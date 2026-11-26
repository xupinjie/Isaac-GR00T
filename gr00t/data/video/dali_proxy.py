"""DALI's standard PyTorch proxy for GR00T's episode-fragment video requests."""

from collections import deque
from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import default_collate, get_worker_info, IterableDataset

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType, VLAStepData


def build_video_pipeline(max_requests: int, device_id: int):
    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def
    def decode():
        path = fn.external_source(name="path", dtype=types.UINT8, ndim=1)
        frame_ids = fn.external_source(name="frame_ids", dtype=types.INT32, ndim=1)
        rgb = fn.decoders.video(fn.io.file.read(path), device="mixed", frames=frame_ids, pad_mode="none")
        # Proxy materializes each output as a dense Torch tensor. Only the
        # number of frames varies for same-resolution camera videos.
        return fn.pad(rgb, device="gpu", axes=(0,), fill_value=0)

    return decode(batch_size=max_requests, device_id=device_id, num_threads=4, prefetch_queue_depth=1)


@dataclass
class _Sample:
    content: VLAStepData
    cameras: dict[str, int]
    position: int
    frame_count: int


@dataclass
class _Shard:
    samples: list[_Sample]
    requests: list[tuple[str, np.ndarray]]

    def __len__(self):
        return len(self.samples)


def _frame_ids(step, deltas, length, allow_padding):
    indices = [int(step) + delta for delta in deltas]
    if allow_padding:
        return [max(0, min(i, length - 1)) for i in indices]
    # Match iloc negative indexing in the original sample extraction.
    indices = [i + length if i < 0 else i for i in indices]
    if not all(0 <= i < length for i in indices):
        raise IndexError(f"Video indices out of range: {indices}, episode length {length}")
    return indices


def _describe_shard(dataset, shard_index):
    requests, samples = [], []
    modalities = {key: config for key, config in dataset.modality_configs.items() if key != "video"}
    deltas = dataset.modality_configs["video"].delta_indices
    for ep_idx, steps in dataset.sharded_episodes[shard_index]:
        if len(steps) == 0:
            continue
        episode_id, metadata = dataset.episode_loader.load_episode_metadata(ep_idx)
        paths = dataset.episode_loader.get_video_paths(episode_id)
        frames = np.asarray(
            [_frame_ids(step, deltas, len(metadata), dataset.allow_padding) for step in steps],
            dtype=np.int32,
        ).reshape(-1)
        cameras = {}
        for camera, path in paths.items():
            cameras[camera] = len(requests)
            requests.append((path, frames))
        for offset, step in enumerate(steps):
            content = extract_step_data(
                metadata, int(step), modalities, dataset.embodiment_tag, dataset.allow_padding
            )
            samples.append(_Sample(content, cameras, offset * len(deltas), len(deltas)))
    if not requests or not samples:
        raise ValueError("DALI Proxy requires a nonempty shard with video modalities")
    return _Shard(samples, requests)


class _ProxyShards(IterableDataset):
    def __init__(self, dataset, proxy):
        super().__init__()
        self.dataset = dataset
        self.proxy = proxy

    def __iter__(self):
        worker_id = get_worker_info().id
        for shard, order in self.dataset.iter_cached_shards(_describe_shard):
            refs = [
                self.proxy(path=np.frombuffer(path.encode("utf-8"), dtype=np.uint8), frame_ids=frames)
                for path, frames in shard.requests
            ]
            # Submit the existing shard's camera requests using Proxy's registered collator.
            yield worker_id, [shard.samples[i] for i in order], default_collate(refs)


def _one_shard(items):
    (shard,) = items
    return shard


def _decoded_shards_in_train_order(shards, num_workers, batch_size):
    """Match the original DataLoader's round-robin batches of steps per worker."""
    pending = [deque() for _ in range(num_workers)]
    for worker in cycle(range(num_workers)):
        while len(pending[worker]) < batch_size:
            worker_id, samples, rgb = next(shards)
            if not rgb.is_cuda:
                raise RuntimeError("DALI Proxy did not return NVDEC output on CUDA")
            images = rgb.cpu().numpy()
            del rgb
            for sample in samples:
                start = sample.position
                stop = start + sample.frame_count
                sample.content.images = {
                    camera: list(images[index, start:stop])
                    for camera, index in sample.cameras.items()
                }
                pending[worker_id].append(sample.content)
        yield [pending[worker].popleft() for _ in range(batch_size)]


class DaliTrainingLoader:
    def __init__(self, dataset, *, device, batch_size, num_workers, collate_fn):
        try:
            from nvidia.dali.plugin.pytorch.experimental import proxy
        except ImportError as exc:
            raise ImportError("dali_proxy requires NVIDIA DALI with CUDA support (the dali extra)") from exc
        if device.type != "cuda":
            raise ValueError("dali_proxy requires CUDA")
        if num_workers < 1:
            raise ValueError("dali_proxy requires at least one DataLoader worker")
        if torch.multiprocessing.get_start_method() != "fork":
            raise ValueError("This DALI Proxy adapter requires a fork multiprocessing context")

        max_requests = max(
            len(episodes) * len(ds.modality_configs["video"].modality_keys)
            for ds in dataset.datasets
            for episodes in ds.sharded_episodes
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.processor = dataset.processor
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        self.server = proxy.DALIServer(build_video_pipeline(max_requests, device_id))
        self.dataset = _ProxyShards(dataset, self.server.proxy)
        self.loader = proxy.DataLoader(
            self.server,
            self.dataset,
            batch_size=1,
            collate_fn=_one_shard,
            num_workers=num_workers,
            prefetch_factor=1,
            multiprocessing_context="fork",
        )

    def __iter__(self):
        for batch in _decoded_shards_in_train_order(iter(self.loader), self.num_workers, self.batch_size):
            yield self.collate_fn([
                self.processor([{"type": MessageType.EPISODE_STEP.value, "content": sample}])
                for sample in batch
            ])

    def close(self):
        self.server.stop_thread()
