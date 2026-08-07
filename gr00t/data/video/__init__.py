"""Video loading pipelines used by training datasets."""

from .nvc_gop_pipeline import NVC_GOP_REQUEST_KEY, NvcGopBatchPrefetcher, NvcGopRequest

__all__ = ["NVC_GOP_REQUEST_KEY", "NvcGopBatchPrefetcher", "NvcGopRequest"]
