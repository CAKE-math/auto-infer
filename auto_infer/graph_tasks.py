"""Low-level ACL graph-task capture and update primitives."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphTaskEntry:
    handle: object
    event: object


def _runtime(runtime=None):
    if runtime is not None:
        return runtime
    import torch
    return torch.npu


def capture_graph_task(stream, op, runtime=None) -> GraphTaskEntry:
    """Capture one dynamically updated operation with an event wait."""
    runtime = _runtime(runtime)
    event = runtime.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    runtime.graph_task_group_begin(stream)
    op()
    handle = runtime.graph_task_group_end(stream)
    return GraphTaskEntry(handle=handle, event=event)


def update_graph_task(entry: GraphTaskEntry, stream, op, runtime=None) -> None:
    """Update one captured task and release its graph-side event wait."""
    runtime = _runtime(runtime)
    runtime.graph_task_update_begin(stream, entry.handle)
    op()
    runtime.graph_task_update_end(stream)
    entry.event.record(stream)
