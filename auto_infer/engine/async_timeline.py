"""Opt-in profiler markers for the async submission contract."""
import os
from contextlib import nullcontext


class AsyncTimeline:
    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = os.getenv("AUTO_INFER_ASYNC_TRACE", "") == "1"
        self.enabled = enabled

    def span(self, sequence: int, phase: str):
        if not self.enabled:
            return nullcontext()
        import torch
        return torch.profiler.record_function(
            f"auto_infer.async/{sequence}/{phase}")
