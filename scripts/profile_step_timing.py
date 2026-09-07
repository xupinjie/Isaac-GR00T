#!/usr/bin/env python3
"""Launch finetuning while recording unsmoothed wall time and loss for every step.

This is a benchmark-only entry point.  It monkeypatches ``Gr00tTrainer`` at
runtime, so the exact same timing code can be used with multiple git
worktrees without modifying either implementation under test.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import runpy
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import TrainerCallback

from gr00t.data.dataset import lerobot_episode_loader as episode_loader_module
from gr00t.data.video.nvc_gop_pipeline import (
    NvcGopBatchPrefetcher,
    _NvcGopIterator,
)
from gr00t.experiment.trainer import Gr00tTrainer


_NVC_MODE = os.environ.get("PROFILE_NVC_MODE", "full_pipeline")
if _NVC_MODE not in {"ipc_gop", "sync_pipeline", "decode_overlap", "full_pipeline"}:
    raise ValueError(f"Unknown PROFILE_NVC_MODE={_NVC_MODE}")


if _NVC_MODE == "ipc_gop":
    @dataclass(frozen=True)
    class _IpcGopPayload:
        first_frame_id: int
        gop_len: int
        shm_name: str
        data: np.ndarray


    class _IpcGopStoreAdapter:
        def get_batch(self, refs):
            return [ref.data for ref in refs]

        def cleanup(self):
            return None


    def _materialize_nvc_gop_over_ipc(self, request):
        if request.gop_refs is not None:
            return request
        if self._nvc_gop_decoder is None:
            gpu_id = int((self.video_backend_kwargs or {}).get("gpu_id", 0))
            self._nvc_gop_decoder = episode_loader_module.nvc.CreateGopDecoder(
                maxfiles=max(1, len(request.video_paths)),
                iGpu=gpu_id,
            )

        payloads_by_video = []
        for video_path, frame_ids in zip(request.video_paths, request.frame_ids):
            payloads_by_key = {}
            for frame_id in frame_ids:
                numpy_data, first_frame_ids, gop_lens = self._nvc_gop_decoder.GetGOPList(
                    [video_path], [int(frame_id)], useGOPCache=False
                )[0]
                if len(first_frame_ids) != 1 or len(gop_lens) != 1:
                    raise RuntimeError(
                        "ACCV-Lab GetGOPList returned an unexpected GOP description "
                        f"for {video_path} frame {frame_id}"
                    )
                first_frame_id = int(first_frame_ids[0])
                gop_len = int(gop_lens[0])
                key = (first_frame_id, gop_len)
                payloads_by_key[key] = _IpcGopPayload(
                    first_frame_id=first_frame_id,
                    gop_len=gop_len,
                    shm_name=f"ipc:{video_path}:{first_frame_id}:{gop_len}",
                    data=np.array(numpy_data, dtype=np.uint8, copy=True),
                )
            payloads_by_video.append(
                [payloads_by_key[key] for key in sorted(payloads_by_key)]
            )
        return request.with_gop_refs(payloads_by_video)


    episode_loader_module.LeRobotEpisodeLoader.materialize_nvc_gop_request = (
        _materialize_nvc_gop_over_ipc
    )

    _original_prefetcher_init = NvcGopBatchPrefetcher.__init__

    def _ipc_prefetcher_init(self, *args, **kwargs):
        _original_prefetcher_init(self, *args, **kwargs)
        self._store.cleanup()
        self.dataset.clear_nvc_gop_store_configuration()
        self._store = _IpcGopStoreAdapter()
        logging.info(
            "Benchmark GOP transport: raw NumPy GOP payloads through DataLoader IPC; "
            "SharedGopStore and GOP caches disabled"
        )

    NvcGopBatchPrefetcher.__init__ = _ipc_prefetcher_init
elif _NVC_MODE == "sync_pipeline":
    # Leave submission to the next iterator call, where it is immediately awaited.
    NvcGopBatchPrefetcher.start_prefetch_after_forward = lambda self: None
elif _NVC_MODE == "decode_overlap":
    # Decode in the background, but return owned frames to the training thread
    # before applying image transforms.
    def _defer_transform(self, pending, samples):
        return pending, samples, None

    NvcGopBatchPrefetcher._transform_decoded = _defer_transform

    def _decode_overlap_next(self):
        self.start_prefetch()
        result = self.worker.wait()
        if result is None:
            self.owner._active_iterator = False
            raise StopIteration
        pending, decoded_samples, ready_event = result
        assert ready_event is None
        transformed_samples = self.owner.processor.transform_decoded_vlm_images(decoded_samples)
        batch = self.owner._collate_transformed(pending, transformed_samples)
        self._awaiting_submit = True
        return batch

    _NvcGopIterator.__next__ = _decode_overlap_next


class _StepRecorder:
    def __init__(self, trainer: Gr00tTrainer) -> None:
        self.trainer = trainer
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.variant = os.environ.get("PROFILE_VARIANT", "unknown")
        self.output_dir = Path(os.environ["PROFILE_TIMING_DIR"])
        self.records: list[dict[str, Any]] = []
        self._data_wait_s: float | None = None
        self._batch_ready_ns: int | None = None
        self._written = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.output_dir / f"rank_{self.rank:02d}.jsonl"
        self._jsonl = self._jsonl_path.open("w", buffering=1)

    def batch_ready(self, wait_start_ns: int, ready_ns: int) -> None:
        self._data_wait_s = (ready_ns - wait_start_ns) / 1e9
        self._batch_ready_ns = ready_ns

    def step_end(self, step: int) -> None:
        end_ns = time.perf_counter_ns()
        if self._data_wait_s is None or self._batch_ready_ns is None:
            return

        # Read loss after taking the time stamp so the benchmark bookkeeping
        # itself is not charged to model training.
        loss = getattr(self.trainer, "loss", None)
        loss_value = float(loss.detach().float().item()) if loss is not None else None
        train_s = (end_ns - self._batch_ready_ns) / 1e9
        record = {
            "step": int(step),
            "data_wait_s": self._data_wait_s,
            "train_s": train_s,
            "total_s": self._data_wait_s + train_s,
            "loss": loss_value,
        }
        self.records.append(record)
        self._jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
        if step % 10 == 0:
            self._jsonl.flush()
        self._data_wait_s = None
        self._batch_ready_ns = None

    def write(self) -> None:
        if self._written:
            return
        self._jsonl.flush()
        self._jsonl.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": self.variant,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "host": socket.gethostname(),
            "clock": "time.perf_counter_ns",
            "records": self.records,
        }
        target = self.output_dir / f"rank_{self.rank:02d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(target)
        self._written = True


class _TimingIterator:
    def __init__(
        self, iterator: Any, recorder: _StepRecorder, first_wait_start_ns: int | None = None
    ) -> None:
        self._iterator = iterator
        self._recorder = recorder
        self._first_wait_start_ns = first_wait_start_ns

    def __iter__(self) -> "_TimingIterator":
        return self

    def __next__(self) -> Any:
        wait_start_ns = self._first_wait_start_ns or time.perf_counter_ns()
        self._first_wait_start_ns = None
        value = next(self._iterator)
        self._recorder.batch_ready(wait_start_ns, time.perf_counter_ns())
        return value


class _TimingLoader:
    def __init__(self, loader: Any, recorder: _StepRecorder) -> None:
        self._loader = loader
        self._recorder = recorder

    def __iter__(self) -> _TimingIterator:
        # Include worker creation (notably expensive with spawn) in the first
        # step's data wait, matching the wall-clock boundary of a normal loop.
        wait_start_ns = time.perf_counter_ns()
        iterator = iter(self._loader)
        return _TimingIterator(iterator, self._recorder, wait_start_ns)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _TimingCallback(TrainerCallback):
    def __init__(self, recorder: _StepRecorder) -> None:
        self.recorder = recorder

    def on_step_end(self, args, state, control, **kwargs):
        self.recorder.step_end(state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.recorder.write()
        return control


_original_init = Gr00tTrainer.__init__
_original_get_train_dataloader = Gr00tTrainer.get_train_dataloader
_original_train = Gr00tTrainer.train


def _profiled_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._step_recorder = _StepRecorder(self)
    self.add_callback(_TimingCallback(self._step_recorder))


def _profiled_get_train_dataloader(self):
    loader = _original_get_train_dataloader(self)
    return _TimingLoader(loader, self._step_recorder)


def _profiled_train(self, *args, **kwargs):
    try:
        return _original_train(self, *args, **kwargs)
    finally:
        self._step_recorder.write()


def _skip_final_model_save(self, *args, **kwargs):
    # The benchmark has no use for a ~40 GB final checkpoint.
    return None


Gr00tTrainer.__init__ = _profiled_init
Gr00tTrainer.get_train_dataloader = _profiled_get_train_dataloader
Gr00tTrainer.train = _profiled_train
Gr00tTrainer.save_model = _skip_final_model_save
Gr00tTrainer._save_checkpoint = _skip_final_model_save


if __name__ == "__main__":
    runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")
