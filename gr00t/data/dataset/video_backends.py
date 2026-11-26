"""Names and capabilities of video decoders used by the training data pipeline."""

STANDARD_VIDEO_BACKENDS = frozenset({"torchcodec", "decord", "ffmpeg", "opencv"})
ON_DEMAND_VIDEO_BACKENDS = frozenset({"accvlab", "decord_gpu", "pynvvideocodec"})

_VIDEO_BACKEND_ALIASES = {
    "accv-lab": "accvlab",
    "decord-gpu": "decord_gpu",
    "decord_cuda": "decord_gpu",
    "nvc": "accvlab",
    "pynv": "pynvvideocodec",
}


def normalize_video_backend(video_backend: str) -> str:
    """Return the canonical, case-insensitive name for a video backend."""
    name = video_backend.strip().lower()
    return _VIDEO_BACKEND_ALIASES.get(name, name)


def is_on_demand_video_backend(video_backend: str) -> bool:
    """Whether the backend decodes only the sampled frame indices."""
    return normalize_video_backend(video_backend) in ON_DEMAND_VIDEO_BACKENDS


def multiprocessing_context_for(video_backend: str) -> str:
    """Return the safe DataLoader multiprocessing context for the backend."""
    return "spawn" if is_on_demand_video_backend(video_backend) else "fork"


def supported_video_backends() -> tuple[str, ...]:
    """Return canonical backend names accepted by the data pipeline."""
    return tuple(sorted(STANDARD_VIDEO_BACKENDS | ON_DEMAND_VIDEO_BACKENDS))
