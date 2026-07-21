"""Host tests for the serving path (no NPU): EngineService's background step
loop batches concurrent submits, recovers from a crashing step instead of
hanging clients, and EngineCore.abort frees an in-flight request. Uses
MockExecutor (deterministic: next = (last + 1) % vocab)."""
import time
import threading
import pytest

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.engine_core import EngineCore
from auto_infer.engine.executor import Executor, MockExecutor
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.serving.service import EngineService


def _cfg(num_blocks=100):
    return EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
    )


def _sp(max_tokens):
    return SamplingParams(max_tokens=max_tokens)


def _drain(q, timeout=5.0):
    """Collect tokens until the None sentinel. Raises (queue.Empty) if the stream
    hangs past `timeout` — that failure IS the 'must not hang' assertion."""
    out = []
    deadline = time.monotonic() + timeout
    while True:
        t = q.get(timeout=max(0.01, deadline - time.monotonic()))
        if t is None:
            return out
        out.append(t)


def test_serving_engine_single_request():
    eng = EngineService(_cfg(), MockExecutor(vocab_size=1000))
    rid, q = eng.submit([1, 2, 3], _sp(3))
    assert _drain(q) == [4, 5, 6]           # last prompt token 3 -> 4, 5, 6
    eng.release(rid)
    eng.close()


def test_serving_engine_continuous_batching():
    # Several concurrent submits are batched by the ONE background loop; each
    # stream is independent and deterministic.
    eng = EngineService(_cfg(), MockExecutor(vocab_size=1000))
    cases = [([1, 2, 3], [4, 5, 6, 7]),
             ([10, 11], [12, 13, 14, 15]),
             ([20], [21, 22, 23, 24])]
    handles = [(eng.submit(ids, _sp(4)), expect) for ids, expect in cases]
    for (rid, q), expect in handles:
        assert _drain(q) == expect
        eng.release(rid)
    eng.close()


def test_serving_engine_never_publishes_async_placeholders():
    config = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
        async_scheduling=True,
        async_batches=2,
    )
    service = EngineService(config, MockExecutor(vocab_size=1000))

    rid, stream = service.submit([1, 2, 3], _sp(4))

    assert _drain(stream) == [4, 5, 6, 7]
    service.release(rid)
    service.close()


def test_late_request_joins_an_active_decode_batch():
    class RecordingExecutor(MockExecutor):
        def __init__(self):
            super().__init__(vocab_size=1000)
            self.batches = []

        def submit(self, plan, prev_sampled=None):
            self.batches.append([
                (item.request_id, item.is_prefill) for item in plan.scheduled])
            return super().submit(plan, prev_sampled)

    executor = RecordingExecutor()
    core = EngineCore(_cfg(), executor)
    first = Request("first", [1, 2, 3], _sp(4))
    late = Request("late", [10, 11], _sp(3))
    core.add_request(first)
    core.step()  # first is now decoding, with one output token

    core.add_request(late)
    core.step()

    assert executor.batches[-1] == [("first", False), ("late", True)]
    assert first.output_token_ids == [4, 5]
    assert late.output_token_ids == [12]
    while core.has_unfinished():
        core.step()
    assert first.output_token_ids == [4, 5, 6, 7]
    assert late.output_token_ids == [12, 13, 14]


def test_serving_engine_recovers_from_bad_step():
    # A step that raises must not hang clients: _recover ends every in-flight
    # stream with the sentinel and rebuilds the engine so new requests still work.
    class BoomExecutor(MockExecutor):
        def __init__(self):
            super().__init__(vocab_size=1000)
            self.boom = True

        def execute(self, plan):
            if self.boom:
                raise RuntimeError("boom")
            return super().execute(plan)

    executor = BoomExecutor()
    eng = EngineService(_cfg(), executor)
    rid, q = eng.submit([1, 2, 3], _sp(3))
    with pytest.raises(RuntimeError, match="engine request failed: boom"):
        _drain(q)
    eng.release(rid)
    executor.boom = False
    rid2, q2 = eng.submit([1, 2, 3], _sp(3))
    assert _drain(q2) == [4, 5, 6]          # engine recovered, serves new requests
    eng.release(rid2)
    eng.close()


def test_nonrecoverable_failure_closes_admission_and_executor():
    class FatalExecutor(Executor):
        def __init__(self):
            self.closed = False

        def execute(self, plan):
            raise RuntimeError("fatal")

        def close(self):
            self.closed = True

    executor = FatalExecutor()
    eng = EngineService(_cfg(), executor)
    _, stream = eng.submit([1], _sp(1))
    with pytest.raises(RuntimeError, match="fatal"):
        stream.get(timeout=5)
    with pytest.raises(RuntimeError, match="closed"):
        eng.submit([2], _sp(1))
    eng.close()
    assert executor.closed


def test_engine_core_abort_frees_request():
    # abort() drops all engine-side state for a live request (client disconnected).
    eng = EngineCore(_cfg(), MockExecutor(vocab_size=1000))
    eng.add_request(Request("x", [1, 2, 3], _sp(100)))
    eng.step()                              # prefill + first token; request is live
    assert eng.has_unfinished()
    eng.abort("x")
    assert not eng.has_unfinished()         # no running / waiting / in-flight left
    assert "x" not in eng.scheduler.block_tables
    assert "x" not in eng.scheduler._requests


def test_engine_core_abort_unknown_is_safe():
    eng = EngineCore(_cfg(), MockExecutor(vocab_size=1000))
    eng.abort("never-existed")              # must not raise


def test_close_uses_executor_close_to_unblock_a_stuck_step():
    class BlockingExecutor(Executor):
        def __init__(self):
            self.entered = threading.Event()
            self.unblock = threading.Event()
            self.closed = False

        def execute(self, plan):
            self.entered.set()
            self.unblock.wait()
            raise RuntimeError("closed")

        def close(self):
            self.closed = True
            self.unblock.set()

    executor = BlockingExecutor()
    service = EngineService(_cfg(), executor, close_timeout_s=0.01)
    service.submit([1], _sp(1))
    assert executor.entered.wait(timeout=1)

    service.close()

    assert executor.closed
    assert not service.thread.is_alive()


def test_submission_queue_is_bounded_and_queued_abort_is_not_executed():
    class BlockingFirstExecutor(MockExecutor):
        def __init__(self):
            super().__init__(vocab_size=1000)
            self.entered = threading.Event()
            self.unblock = threading.Event()
            self.prefill_ids = []

        def execute(self, plan):
            self.prefill_ids.extend(
                item.request_id for item in plan.scheduled if item.is_prefill
            )
            if not self.entered.is_set():
                self.entered.set()
                self.unblock.wait()
            return super().execute(plan)

    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64),
    )
    executor = BlockingFirstExecutor()
    service = EngineService(cfg, executor)
    first, first_stream = service.submit([1], _sp(1))
    assert executor.entered.wait(timeout=1)
    cancelled, cancelled_stream = service.submit([10], _sp(1))
    third, third_stream = service.submit([20], _sp(1))

    with pytest.raises(RuntimeError, match="submission queue is full"):
        service.submit([30], _sp(1))
    service.release(cancelled)
    assert _drain(cancelled_stream) == []
    executor.unblock.set()

    assert _drain(first_stream) == [2]
    assert _drain(third_stream) == [21]
    service.release(first)
    service.release(third)
    service.close()
    assert cancelled not in executor.prefill_ids


def test_close_is_bounded_when_executor_close_also_hangs():
    class FullyStuckExecutor(Executor):
        def __init__(self):
            self.execute_entered = threading.Event()
            self.release = threading.Event()

        def execute(self, plan):
            self.execute_entered.set()
            self.release.wait()
            raise RuntimeError("released")

        def close(self):
            self.release.wait()

    executor = FullyStuckExecutor()
    service = EngineService(_cfg(), executor, close_timeout_s=0.01)
    service.submit([1], _sp(1))
    assert executor.execute_entered.wait(timeout=1)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="executor close timed out"):
        service.close()

    assert time.monotonic() - started < 0.2
    executor.release.set()
