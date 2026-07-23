"""Opt-in profiler markers for the async submission contract."""
import os
import json
from pathlib import Path
from threading import Lock
from contextlib import nullcontext

_events = []
_event_lock = Lock()


def clear_async_events():
    with _event_lock:
        _events.clear()


def record_async_interval(sequence, phase, start_ns, end_ns):
    with _event_lock:
        _events.append({
            "name": f"auto_infer.async/{sequence}/{phase}",
            "cat": "auto_infer.async",
            "ph": "X",
            "ts": start_ns / 1000,
            "dur": (end_ns - start_ns) / 1000,
            "pid": "AUTO INFER ASYNC",
            "tid": "graph-task-update",
        })


def add_async_events_to_trace(path):
    with _event_lock:
        events = list(_events)
    trace_path = Path(path)
    payload = json.loads(trace_path.read_text())
    target = payload["traceEvents"] if isinstance(payload, dict) else payload
    target.extend(events)
    trace_path.write_text(json.dumps(payload, separators=(",", ":")))


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
