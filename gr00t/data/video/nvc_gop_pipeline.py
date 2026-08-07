"""Deferred ACCV-Lab GOP loading and main-process asynchronous decode.

The data loader workers only demux compressed GOP byte ranges and put them in
``SharedGopStore``. This module resolves the references in the training process,
submits a whole batch to NVDEC, and runs GPU image transforms.

The iterator intentionally owns two buffers: decoded frames from batch N are
cloned out of ACCV-Lab's reusable output pool before batch N+1 is submitted.
That lets NVDEC for N+1 overlap the GPU image transforms and model work for N.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import os
import threading
from typing import Any, Iterator

import torch


NVC_GOP_REQUEST_KEY = "_nvc_gop_request"


@dataclass(frozen=True)
class NvcGopRequest:
    """One sample's videos, requested frames, and optional shared-memory refs.

    ``video_paths`` and ``frame_ids`` are ordered by camera view. ``gop_refs``
    is populated lazily in a DataLoader worker immediately before collation.
    Keeping the preloaded shard free of GOP references bounds the number of
    live shared-memory entries by the DataLoader queue depth instead of by the
    much larger shard size.
    """

    video_paths: tuple[str, ...]
    frame_ids: tuple[tuple[int, ...], ...]
    image_keys: tuple[str, ...]
    language: str
    gop_refs: tuple[tuple[Any, ...], ...] | None = None

    def with_gop_refs(self, gop_refs: list[list[Any]]) -> "NvcGopRequest":
        return replace(self, gop_refs=tuple(tuple(refs) for refs in gop_refs))


@dataclass
class _VideoOccurrence:
    group_index: int
    frame_positions: tuple[int, ...]


@dataclass
class _PendingDecode:
    batch: Any
    requests: list[NvcGopRequest]
    numpy_datas: list[list[Any]]
    video_paths: list[str]
    frame_ids: list[list[int]]
    sample_occurrences: list[list[_VideoOccurrence]]


def make_store_id(global_rank: int) -> int:
    """Return a job- and process-specific numeric SharedGopStore identifier."""

    job_id = os.environ.get("SLURM_JOB_ID")
    job_component = int(job_id) if job_id and job_id.isdigit() else os.getpid()
    # Rank is included because every rank owns an independent DataLoader and
    # decoder. PID protects concurrent non-Slurm runs on the same host.
    return job_component * 100_000 + os.getpid() % 10_000 * 10 + global_rank


def calculate_store_capacity(
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_videos_per_sample: int,
    max_frames_per_video: int,
) -> int:
    """Size the store so no reference still in the DataLoader queue is evicted."""

    # DataLoader keeps ``num_workers * prefetch_factor`` tasks outstanding,
    # while each worker may already be materializing its next batch and the
    # main process owns the current/next decode batches.  Count those worker-
    # local producers explicitly; otherwise a fast worker can evict a GOP
    # referenced by an older ordered batch before the main process resolves it.
    queued_batches = max(1, num_workers * prefetch_factor) + num_workers + 2
    refs_per_batch = batch_size * max_videos_per_sample * max_frames_per_video
    return max(1, queued_batches * refs_per_batch)


def _unlink_store_orphans(store_id: int) -> int:
    """Remove evicted GOP blocks left behind by ACCV-Lab's native store.

    ``SharedGopStore.cleanup()`` releases the entries that remain indexed, but
    older native builds do not unlink blocks that were evicted from the index.
    All workers are stopped before this helper runs, so files under the exact
    per-rank store prefix are no longer in use.
    """

    prefix = f"gs_{int(store_id)}_"
    removed = 0
    try:
        entries = os.scandir("/dev/shm")
    except FileNotFoundError:
        return 0
    with entries:
        for entry in entries:
            if not entry.name.startswith(prefix):
                continue
            try:
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                pass
    return removed


class NvcGopBatchPrefetcher:
    """Wrap a DataLoader with ACCV-Lab GOP resolution and double buffering."""

    def __init__(
        self,
        dataloader,
        *,
        dataset,
        processor,
        batch_size: int,
        num_workers: int,
        device: torch.device | str | None = None,
    ) -> None:
        try:
            import accvlab.on_demand_video_decoder as nvc
        except ImportError as exc:  # pragma: no cover - exercised in ACCV environment
            raise ImportError(
                "The nvc backend requires accvlab.on_demand_video_decoder."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("The nvc GOP pipeline requires a CUDA device")

        self.dataloader = dataloader
        self.dataset = dataset
        self.processor = processor
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(getattr(dataloader, "prefetch_factor", None) or 2)
        self.device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        self.gpu_id = self.device.index if self.device.index is not None else 0
        self._nvc = nvc

        max_videos, max_frames = dataset.get_nvc_gop_shape()
        self.max_videos_per_sample = max_videos
        self.max_frames_per_video = max_frames
        self.max_grouped_frames = max(2, self.max_frames_per_video)
        self.store_capacity = calculate_store_capacity(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            max_videos_per_sample=max_videos,
            max_frames_per_video=max_frames,
        )

        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.store_id = make_store_id(global_rank)
        self.dataset.configure_nvc_gop_store(
            store_id=self.store_id,
            capacity=self.store_capacity,
        )
        # This must happen before iter(dataloader), which starts worker processes.
        self._store = nvc.SharedGopStore.create(
            capacity=self.store_capacity,
            store_id=self.store_id,
        )
        # GPU objects are created lazily after iter(dataloader) starts forked
        # workers. This keeps CUDA/NVDEC state exclusively in the main process.
        self._decoder = None
        self._copy_stream = None
        self._transform_stream = None
        self._closed = False
        self._active_iterator = False
        self._iterator = None

        logging.info(
            "Configured nvc GOP pipeline: store_id=%s capacity=%s maxfiles=%s "
            "max_frames=%s grouped_frames=%s device=%s",
            self.store_id,
            self.store_capacity,
            self.batch_size * self.max_videos_per_sample,
            self.max_frames_per_video,
            self.max_grouped_frames,
            self.device,
        )

    def _initialize_gpu_pipeline(self) -> None:
        if self._decoder is not None:
            return
        self._decoder = self._nvc.CreateBatchAsyncGopDecoder(
            maxfiles=self.batch_size * self.max_videos_per_sample,
            max_frames_per_decode_call=self.max_grouped_frames,
            iGpu=self.gpu_id,
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._transform_stream = torch.cuda.Stream(device=self.device)

    def __getattr__(self, name: str):
        # Preserve DataLoader attributes used by Trainer/Accelerate.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dataloader, name)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("NvcGopBatchPrefetcher has already been closed")
        if self._active_iterator:
            raise RuntimeError("Only one nvc GOP prefetch iterator may be active")
        self._active_iterator = True
        self._iterator = _NvcGopIterator(self)
        return self._iterator

    def start_prefetch_after_forward(self) -> None:
        """Start N+1 once Trainer has finished forward(N).

        Delaying GPU decode and transforms until this point makes their main
        overlap window backward/optimizer instead of competing with the model's
        forward pass. Iterators used outside ``Gr00tTrainer`` remain functional:
        their next ``__next__`` call starts any deferred work synchronously.
        """

        if self._active_iterator and self._iterator is not None:
            self._iterator.start_prefetch()

    def _submit(self, batch: Any) -> _PendingDecode:
        if self._decoder is None:
            raise RuntimeError("nvc GPU decoder has not been initialized")
        inputs = batch["inputs"]
        requests = inputs.get(NVC_GOP_REQUEST_KEY)
        if not requests:
            raise RuntimeError(f"Raw nvc batch is missing {NVC_GOP_REQUEST_KEY}")
        if not all(isinstance(request, NvcGopRequest) for request in requests):
            raise TypeError("Invalid nvc GOP request in collated batch")
        if not all(request.gop_refs is not None for request in requests):
            raise RuntimeError("DataLoader worker returned unresolved GOP references")

        # Group repeated camera files across samples. Sharded batches commonly
        # contain adjacent steps from one episode, so small chunks reuse GOP
        # decode work without collapsing the batch to too few NVDEC streams.
        path_to_group: dict[str, int] = {}
        video_paths: list[str] = []
        frame_ids: list[list[int]] = []
        refs_by_group: list[dict[tuple[int, int, str], Any]] = []
        sample_occurrences: list[list[_VideoOccurrence]] = []
        for request in requests:
            assert request.gop_refs is not None
            if not (
                len(request.video_paths)
                == len(request.frame_ids)
                == len(request.gop_refs)
                == len(request.image_keys)
            ):
                raise RuntimeError("Inconsistent camera dimensions in nvc GOP request")
            occurrences: list[_VideoOccurrence] = []
            for path, ids, refs in zip(
                request.video_paths, request.frame_ids, request.gop_refs
            ):
                if not ids:
                    raise RuntimeError("nvc GOP request contains an empty frame list")
                group_index = path_to_group.get(path)
                if (
                    group_index is None
                    or len(frame_ids[group_index]) + len(ids) > self.max_grouped_frames
                ):
                    group_index = len(video_paths)
                    path_to_group[path] = group_index
                    video_paths.append(path)
                    frame_ids.append([])
                    refs_by_group.append({})
                positions = tuple(
                    range(
                        len(frame_ids[group_index]),
                        len(frame_ids[group_index]) + len(ids),
                    )
                )
                frame_ids[group_index].extend(int(frame_id) for frame_id in ids)
                for ref in refs:
                    refs_by_group[group_index][
                        (int(ref.first_frame_id), int(ref.gop_len), ref.shm_name)
                    ] = ref
                occurrences.append(
                    _VideoOccurrence(
                        group_index=group_index,
                        frame_positions=positions,
                    )
                )
            sample_occurrences.append(occurrences)

        frames_per_video = [len(ids) for ids in frame_ids]
        max_frames = max(frames_per_video)
        decoder_frame_limit = self.max_grouped_frames
        if max_frames > decoder_frame_limit:
            raise RuntimeError(
                f"Grouped batch requests {max_frames} frames/video, decoder limit is "
                f"{decoder_frame_limit}"
            )
        # ACCV-Lab requires rectangular [video][frame] input. Repeat the last
        # ID for shorter groups and ignore those padding positions on collect.
        padded_frame_ids = [
            ids + [ids[-1]] * (max_frames - len(ids)) for ids in frame_ids
        ]

        flat_refs: list[Any] = []
        ref_counts: list[int] = []
        for refs in refs_by_group:
            ordered_refs = [refs[key] for key in sorted(refs)]
            flat_refs.extend(ordered_refs)
            ref_counts.append(len(ordered_refs))

        arrays = self._store.get_batch(flat_refs)
        numpy_datas: list[list[Any]] = []
        offset = 0
        for count in ref_counts:
            numpy_datas.append(arrays[offset : offset + count])
            offset += count

        self._decoder.DecodeFromGOPListRGB(
            numpy_datas,
            video_paths,
            padded_frame_ids,
            False,
        )
        return _PendingDecode(
            batch=batch,
            requests=requests,
            numpy_datas=numpy_datas,
            video_paths=video_paths,
            frame_ids=padded_frame_ids,
            sample_occurrences=sample_occurrences,
        )

    def _collect_and_clone(self, pending: _PendingDecode) -> list[torch.Tensor]:
        if self._decoder is None or self._copy_stream is None:
            raise RuntimeError("nvc GPU decoder has not been initialized")
        decoded = self._decoder.DecodeFromGOPListRGBGetBuffer(
            pending.video_paths,
            pending.frame_ids,
            False,
        )
        if len(decoded) != len(pending.video_paths):
            raise RuntimeError(
                f"ACCV-Lab returned {len(decoded)} video rows for "
                f"{len(pending.video_paths)} requests"
            )

        # ``decoded`` points into ACCV-Lab's reusable output pool, so the final
        # sample tensors must own their storage before the next decode submit.
        samples: list[torch.Tensor] = []
        with torch.cuda.stream(self._copy_stream):
            for occurrences in pending.sample_occurrences:
                if not occurrences:
                    raise RuntimeError("nvc GOP request contains no camera views")
                temporal_length = len(occurrences[0].frame_positions)
                if not all(
                    len(occurrence.frame_positions) == temporal_length
                    for occurrence in occurrences
                ):
                    raise RuntimeError(
                        "Camera views in one sample have different temporal lengths"
                    )
                for occurrence in occurrences:
                    row = decoded[occurrence.group_index]
                    if not occurrence.frame_positions:
                        raise RuntimeError("nvc GOP request contains no frame positions")
                    last_position = occurrence.frame_positions[-1]
                    if len(row) <= last_position:
                        raise RuntimeError(
                            f"ACCV-Lab returned {len(row)} frames for video "
                            f"{occurrence.group_index}; expected position "
                            f"{last_position}"
                        )
                # Match the legacy processor order: [T, V, C, H, W] ->
                # [T*V, C, H, W]. torch.stack owns the result, so no separate
                # clone of each decoder-backed frame is necessary.
                sample_frames = [
                    torch.as_tensor(
                        decoded[occurrence.group_index][
                            occurrence.frame_positions[time]
                        ],
                        device=self.device,
                    ).permute(2, 0, 1)
                    for time in range(temporal_length)
                    for occurrence in occurrences
                ]
                samples.append(torch.stack(sample_frames, dim=0))

        # ACCV-Lab owns and reuses ``decoded``. Its next submit runs on an
        # internal stream that cannot wait on a PyTorch event, so only the
        # small D2D ownership copy is synchronized here. Image transforms stay
        # asynchronous and overlap the next NVDEC submission.
        self._copy_stream.synchronize()

        return samples

    def _transform_decoded(
        self, pending: _PendingDecode, samples: list[torch.Tensor]
    ) -> tuple[_PendingDecode, list[torch.Tensor], torch.cuda.Event]:
        if self._transform_stream is None:
            raise RuntimeError("nvc transform stream has not been initialized")
        with torch.cuda.stream(self._transform_stream):
            transformed_samples = self.processor.transform_decoded_vlm_images(
                decoded_samples=samples
            )
            ready_event = torch.cuda.Event()
            ready_event.record(self._transform_stream)
        return pending, transformed_samples, ready_event

    def _collate_transformed(
        self,
        pending: _PendingDecode,
        transformed_samples: list[torch.Tensor],
    ) -> Any:
        inputs = pending.batch["inputs"]
        inputs.pop(NVC_GOP_REQUEST_KEY)
        vlm_inputs = self.processor.collate_transformed_vlm_inputs(
            transformed_samples=transformed_samples,
            languages=[request.language for request in pending.requests],
            device=self.device,
        )
        inputs.update(vlm_inputs)
        return pending.batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._iterator is not None:
            try:
                self._iterator.close()
            except BaseException:
                logging.exception("Failed while draining nvc background prefetch during cleanup")
            self._iterator = None
        try:
            if self._decoder is not None:
                release_memory = getattr(self._decoder, "release_device_memory", None)
                if release_memory is not None:
                    release_memory()
                release_decoder = getattr(self._decoder, "release_decoder", None)
                if release_decoder is not None:
                    release_decoder()
        finally:
            self._decoder = None
            if self._store is not None:
                self._store.cleanup()
                self._store = None
            orphan_count = _unlink_store_orphans(self.store_id)
            self.dataset.clear_nvc_gop_store_configuration()
            logging.info(
                "Closed nvc GOP pipeline (unlinked_orphans=%s)",
                orphan_count,
            )


class _NvcGopIterator:
    def __init__(self, owner: NvcGopBatchPrefetcher) -> None:
        self.owner = owner
        self.raw_iterator = iter(owner.dataloader)
        owner._initialize_gpu_pipeline()
        self.worker = _DecodeTransformWorker(owner, self.raw_iterator)
        self._awaiting_submit = False
        self.worker.submit()

    def __iter__(self):
        return self

    def __next__(self):
        # Non-Trainer consumers do not have an after-forward hook. Preserve
        # ordinary iterator semantics by starting deferred work here, although
        # those callers naturally do not get backward overlap.
        self.start_prefetch()
        result = self.worker.wait()
        if result is None:
            self.owner._active_iterator = False
            raise StopIteration
        pending, transformed_samples, ready_event = result
        torch.cuda.current_stream(self.owner.device).wait_event(ready_event)
        batch = self.owner._collate_transformed(pending, transformed_samples)
        self._awaiting_submit = True
        return batch

    def start_prefetch(self) -> None:
        if not self._awaiting_submit:
            return
        self._awaiting_submit = False
        self.worker.submit()

    def __del__(self):
        self.close()

    def close(self) -> None:
        try:
            if self.worker.thread is not None:
                self.worker.wait()
        finally:
            shutdown_workers = getattr(self.raw_iterator, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()
            self.owner._active_iterator = False


class _DecodeTransformWorker:
    """One-at-a-time background fetch, NVDEC, clone, and GPU transform."""

    def __init__(self, owner: NvcGopBatchPrefetcher, raw_iterator: Iterator[Any]) -> None:
        self.owner = owner
        self.raw_iterator = raw_iterator
        self.thread: threading.Thread | None = None
        self.result: tuple[_PendingDecode, list[torch.Tensor], torch.cuda.Event] | None = None
        self.exception: BaseException | None = None
        self.exhausted = False

    def submit(self) -> None:
        if self.exhausted:
            return
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Previous nvc background prefetch is still running")
        self.result = None
        self.exception = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.owner.device)
            raw_batch = next(self.raw_iterator)
            pending = self.owner._submit(raw_batch)
            owned_samples = self.owner._collect_and_clone(pending)
            self.result = self.owner._transform_decoded(pending, owned_samples)
        except StopIteration:
            self.exhausted = True
        except BaseException as exc:
            self.exception = exc

    def wait(self) -> tuple[_PendingDecode, list[torch.Tensor], torch.cuda.Event] | None:
        if self.thread is None:
            return None
        self.thread.join()
        self.thread = None
        if self.exception is not None:
            raise self.exception
        return self.result
