"""Decode a single episode fragment across its cameras in a DataLoader worker."""

import os

import numpy as np


def fragment_frame_indices(steps, deltas, length: int, allow_padding: bool) -> np.ndarray:
    positions = np.asarray(steps, dtype=np.int64)[:, None] + np.asarray(deltas, dtype=np.int64)
    if allow_padding:
        positions = np.clip(positions, 0, length - 1)
    # Indexing the frame numbers matches extract_step_data's iloc, including negative deltas.
    return np.unique(np.arange(length, dtype=np.int64)[positions])


class DaliVideoDecoder:
    def __init__(self, num_cameras: int, gpu_id: int | None = None, num_threads: int = 2):
        self.num_cameras = num_cameras
        self.gpu_id = int(os.environ.get("LOCAL_RANK", 0)) if gpu_id is None else gpu_id
        self.num_threads = num_threads
        self._pipe = None

    def __getstate__(self):
        # A spawned worker must never inherit a native pipeline from its parent.
        return {**self.__dict__, "_pipe": None}

    def _build(self):
        try:
            from nvidia.dali import fn, pipeline_def, types
        except ImportError as exc:
            raise ImportError(
                "dali_pipeline requires NVIDIA DALI with CUDA (the dali extra)"
            ) from exc

        @pipeline_def
        def decode():
            paths = fn.external_source(name="paths", dtype=types.UINT8, ndim=1)
            frames = fn.external_source(name="frames", dtype=types.INT32, ndim=1)
            return fn.decoders.video(
                fn.io.file.read(paths), device="mixed", frames=frames, pad_mode="none"
            )

        pipe = decode(
            batch_size=self.num_cameras,
            num_threads=self.num_threads,
            device_id=self.gpu_id,
            prefetch_queue_depth=1,
            exec_pipelined=False,
            exec_async=False,
        )
        pipe.build()
        return pipe

    def read(self, paths: dict[str, str], indices: np.ndarray) -> dict[str, np.ndarray]:
        if not len(indices):
            return {key: np.empty(0, dtype=np.uint8) for key in paths}
        if self._pipe is None:
            self._pipe = self._build()
        from nvidia.dali.backend import TensorListGPU

        ordered, inverse = np.unique(np.asarray(indices, dtype=np.int32), return_inverse=True)
        (gpu,) = self._pipe.run(
            paths=[np.frombuffer(os.fsencode(p), dtype=np.uint8) for p in paths.values()],
            frames=[ordered] * len(paths),
        )
        if not isinstance(gpu, TensorListGPU):
            raise RuntimeError("dali_pipeline expected CUDA video output")
        # as_cpu copies out of DALI's reusable GPU output. Views keep this host buffer alive.
        host = gpu.as_cpu()
        cameras = (
            host.as_array() if host.is_dense_tensor() else [host.at(i) for i in range(len(paths))]
        )
        if np.array_equal(ordered, indices):
            return dict(zip(paths, cameras, strict=True))
        return {key: rgb[inverse] for key, rgb in zip(paths, cameras, strict=True)}
