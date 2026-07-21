"""SP4: the batch-queue async path now carries an output-thread FUTURE (the
D2H/CPU-materialization step) instead of a raw handle, so the engine thread
can schedule/build/dispatch the NEXT batch while a dedicated single-worker
output thread does `.tolist()` for the previous one (matching vLLM's
uniproc `async_output_thread`). This must be a pure timing change: async-ON
and async-OFF must produce IDENTICAL output token streams.

MockExecutor doesn't override collect_async/collect_result, so it exercises
the base `Executor.collect_async` default: a synchronous `collect()` wrapped
in an already-resolved `concurrent.futures.Future` — no real background
thread, so these tests are deterministic on host (no NPU / real threading
needed to prove the queue-entry refactor and output invariance)."""
from concurrent.futures import Future

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.execution import DeviceTokenRef
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.entrypoints.llm import LLM

PROMPTS = [[1, 2, 3], [10, 11, 12, 13], [7, 8], [20, 21, 22, 23, 24]]


def _llm(async_scheduling, num_blocks=100, vocab_size=1000):
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
        async_scheduling=async_scheduling,
    )
    return LLM(cfg, executor=MockExecutor(vocab_size=vocab_size))


def test_async_on_off_identical_output_tokens():
    sync_out = _llm(async_scheduling=False).generate(
        [list(p) for p in PROMPTS], max_tokens=6)
    async_out = _llm(async_scheduling=True).generate(
        [list(p) for p in PROMPTS], max_tokens=6)
    assert async_out == sync_out


def test_async_on_off_identical_single_prompt():
    sync_out = _llm(async_scheduling=False).generate([[5, 6, 7]], max_tokens=4)
    async_out = _llm(async_scheduling=True).generate([[5, 6, 7]], max_tokens=4)
    assert async_out == sync_out


def test_async_on_off_identical_under_preemption_pressure():
    # num_blocks=3 (see test_engine_core.py's test_preemption_matches_
    # unpressured_output for why this is the tightest feasible value): forces
    # real preemption/recompute, exercising the _sampled eviction + decode-
    # splice interaction with the future-backed collect.
    prompts = [[1, 2, 3], [4, 5, 6]]
    sync_out = _llm(async_scheduling=False, num_blocks=3).generate(
        [list(p) for p in prompts], max_tokens=8)
    async_out = _llm(async_scheduling=True, num_blocks=3).generate(
        [list(p) for p in prompts], max_tokens=8)
    assert async_out == sync_out


def test_base_executor_collect_async_returns_resolved_future():
    """The SP4 Executor contract itself: collect_async's default is a real
    (already-resolved) Future; collect_result just unwraps it. No thread is
    spun up for sync-only/mock backends."""
    ex = MockExecutor(vocab_size=100)
    handle = {"sampled": {"r0": 7}}
    fut = ex.collect_async(handle)
    assert isinstance(fut, Future)
    assert fut.done()
    assert ex.collect_result(fut).single_tokens() == {"r0": 7}


def test_engine_core_queue_carries_futures_not_handles():
    """Guards the exact SP4 refactor: _step_async's in-flight queue entries
    are (sched, Future), not (sched, handle)."""
    llm = _llm(async_scheduling=True)
    eng = llm.engine
    eng.add_request(Request(request_id="a", prompt_token_ids=[1, 2, 3],
                            sampling=SamplingParams(max_tokens=5)))
    eng.step()
    assert eng._queue, "queue should have at least one in-flight entry after one step"
    for _, entry in eng._queue:
        assert isinstance(entry, Future)


def test_engine_retains_batch_row_references_not_scalar_tokens():
    llm = _llm(async_scheduling=True)
    eng = llm.engine
    eng.add_request(Request(request_id="a", prompt_token_ids=[1, 2, 3],
                            sampling=SamplingParams(max_tokens=5)))

    eng.step()

    assert eng._sampled
    assert all(isinstance(ref, DeviceTokenRef) for ref in eng._sampled.values())
