from contextlib import nullcontext
from dataclasses import dataclass

from auto_infer.graph_tasks import capture_graph_task, update_graph_task
from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline


@dataclass
class _Context:
    seqlens_kv: list[int]
    cu_seqlens_q: list[int] | None = None


class _Graph:
    def __init__(self, trace):
        self.trace = trace

    def replay(self):
        self.trace.append("replay")


class _Backend:
    def __init__(self, trace):
        self.trace = trace
        self.contexts = []

    def update(self, ctx, stream=None):
        self.contexts.append(ctx)
        self.trace.append(f"update:{stream}")


def test_pipeline_enqueues_replay_before_side_stream_update():
    trace = []
    backend = _Backend(trace)
    pipeline = GraphTaskPipeline(
        backend, update_stream="side",
        stream_context=lambda stream: nullcontext())

    pipeline.replay(_Graph(trace), _Context([7, 11]))

    assert trace == ["replay", "update:side"]
    assert backend.contexts[0].seqlens_kv == [7, 11]


def test_pipeline_rotates_without_mutating_inflight_metadata():
    trace = []
    backend = _Backend(trace)
    pipeline = GraphTaskPipeline(
        backend, update_stream="side", metadata_slots=2,
        stream_context=lambda stream: nullcontext())

    pipeline.replay(_Graph(trace), _Context([7, 11], [3, 5]))
    first = backend.contexts[-1]
    pipeline.replay(_Graph(trace), _Context([8, 12], [4, 6]))
    second = backend.contexts[-1]

    assert first.seqlens_kv == [7, 11]
    assert first.cu_seqlens_q == [3, 5]
    assert second.seqlens_kv == [8, 12]
    assert second.cu_seqlens_q == [4, 6]
    assert first.seqlens_kv is not second.seqlens_kv
    assert first.cu_seqlens_q is not second.cu_seqlens_q


def test_pipeline_rejects_fewer_than_two_metadata_slots():
    backend = _Backend([])

    try:
        GraphTaskPipeline(
            backend, update_stream="side", metadata_slots=1,
            stream_context=lambda stream: nullcontext())
    except ValueError as exc:
        assert "metadata_slots" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


class _Event:
    def __init__(self, trace):
        self.trace = trace

    def wait(self, stream):
        self.trace.append(f"wait:{stream}")

    def reset(self, stream):
        self.trace.append(f"reset:{stream}")

    def record(self, stream):
        self.trace.append(f"record:{stream}")


class _Runtime:
    def __init__(self, trace):
        self.trace = trace

    def ExternalEvent(self):
        self.trace.append("event")
        return _Event(self.trace)

    def graph_task_group_begin(self, stream):
        self.trace.append(f"begin:{stream}")

    def graph_task_group_end(self, stream):
        self.trace.append(f"end:{stream}")
        return "handle"

    def graph_task_update_begin(self, stream, handle):
        self.trace.append(f"update-begin:{stream}:{handle}")

    def graph_task_update_end(self, stream):
        self.trace.append(f"update-end:{stream}")


def test_graph_task_event_is_captured_before_the_dynamic_op():
    trace = []
    runtime = _Runtime(trace)

    entry = capture_graph_task("main", lambda: trace.append("op"), runtime=runtime)

    assert entry.handle == "handle"
    assert trace == [
        "event", "wait:main", "reset:main", "begin:main", "op", "end:main"]


def test_graph_task_update_records_event_after_update_end():
    trace = []
    runtime = _Runtime(trace)
    entry = capture_graph_task("main", lambda: trace.append("capture-op"), runtime=runtime)
    trace.clear()

    update_graph_task(
        entry, "side", lambda: trace.append("update-op"), runtime=runtime)

    assert trace == [
        "update-begin:side:handle", "update-op", "update-end:side", "record:side"]
