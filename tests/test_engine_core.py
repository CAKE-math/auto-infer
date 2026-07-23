import pytest

from auto_infer.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from auto_infer.errors import EngineStalledError, RequestRejectedError
from auto_infer.engine.executor import Executor, MockExecutor
from auto_infer.engine.execution import ExecutionResult
from auto_infer.engine.request import Request, RequestStatus, SamplingParams
from auto_infer.entrypoints.llm import LLM


def make_llm():
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
    )
    return LLM(cfg, executor=MockExecutor(vocab_size=1000))


def test_generate_runs_to_max_tokens():
    llm = make_llm()
    outs = llm.generate([[1, 2, 3]], max_tokens=5)
    assert len(outs) == 1
    assert len(outs[0]) == 5


def test_deterministic_mock_tokens():
    # MockExecutor: next = (last + 1) % vocab. First sampled from last prompt token (3).
    llm = make_llm()
    outs = llm.generate([[1, 2, 3]], max_tokens=3)
    assert outs[0] == [4, 5, 6]


def test_multiple_requests():
    llm = make_llm()
    outs = llm.generate([[1, 2, 3], [10, 11]], max_tokens=2)
    assert outs[0] == [4, 5]
    assert outs[1] == [12, 13]


def _llm(num_blocks, async_scheduling=True):
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
        async_scheduling=async_scheduling,
    )
    return LLM(cfg, executor=MockExecutor(vocab_size=1000))


def _count_preemptions(llm):
    """Instrument scheduler.preempt_one on this LLM's engine to count real
    evictions, so the tight-memory test can assert preemption actually fired
    (not just that the run happened not to need it)."""
    calls = {"n": 0}
    orig = llm.engine.scheduler.preempt_one

    def counted():
        calls["n"] += 1
        return orig()

    llm.engine.scheduler.preempt_one = counted
    return calls


# num_blocks=3 (not 2): with block_size=4, a single request's own KV footprint
# at completion is prompt(3) + max_tokens(8) = 11 tokens = ceil(11/4) = 3
# blocks. num_blocks=2 (2*4=8 slots) is below that floor, so no schedule/
# preempt strategy could ever let *either* request finish - it isn't a "tight"
# scenario, it's an infeasible one. num_blocks=3 (12 slots) is the tightest
# value where a single request can complete, while still being far too little
# for both requests' combined worst case (6 blocks), so real contention (and
# real preemption) is forced between them.
def test_preemption_matches_unpressured_output():
    prompts = [[1, 2, 3], [4, 5, 6]]
    roomy = _llm(num_blocks=100).generate([list(p) for p in prompts], max_tokens=8)
    tight = _llm(num_blocks=3)
    calls = _count_preemptions(tight)
    tight_out = tight.generate([list(p) for p in prompts], max_tokens=8)
    assert tight_out == roomy                   # recompute preserves token stream
    assert calls["n"] > 0                        # preemption actually fired


def test_preemption_sync_path():
    prompts = [[1, 2, 3], [4, 5, 6]]
    roomy = _llm(num_blocks=100, async_scheduling=False).generate(
        [list(p) for p in prompts], max_tokens=8)
    tight = _llm(num_blocks=3, async_scheduling=False)
    calls = _count_preemptions(tight)
    tight_out = tight.generate([list(p) for p in prompts], max_tokens=8)
    assert tight_out == roomy
    assert calls["n"] > 0                        # preemption actually fired


def test_oversized_prompt_fails_instead_of_spinning():
    llm = _llm(num_blocks=1, async_scheduling=False)
    with pytest.raises(RequestRejectedError, match=r"requires 2 KV blocks.*capacity is 1"):
        llm.generate([[1, 2, 3, 4, 5]], max_tokens=1)


def test_request_cannot_exceed_configured_model_length():
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock", max_model_len=4),
        cache=CacheConfig(block_size=4, num_blocks=100))
    llm = LLM(cfg, executor=MockExecutor())

    with pytest.raises(RequestRejectedError, match=r"length 5.*max_model_len is 4"):
        llm.generate([[1, 2, 3]], max_tokens=2)


def test_executor_missing_required_sample_fails_instead_of_spinning():
    class BrokenExecutor(Executor):
        def execute(self, plan):
            return ExecutionResult()

    engine = LLM(EngineConfig(model=ModelConfig("/mock")), BrokenExecutor()).engine
    engine.add_request(Request("r", [1], SamplingParams(max_tokens=1)))
    with pytest.raises(EngineStalledError, match="did not return a token"):
        engine.step()


def test_executor_structured_error_is_not_ignored():
    class FailedExecutor(Executor):
        def execute(self, plan):
            return ExecutionResult(errors={"r": "device failure"})

    engine = LLM(EngineConfig(model=ModelConfig("/mock")), FailedExecutor()).engine
    engine.add_request(Request("r", [1], SamplingParams(max_tokens=1)))
    with pytest.raises(RuntimeError, match="device failure"):
        engine.step()


# T9b: EngineCore._sampled must be MERGED (not replaced) on every _step_async
# submit. A running decode request that the scheduler's decode loop doesn't
# schedule for a given batch (num_seqs >= max_num_seqs, or budget < 1) is not
# preempted - it stays in scheduler.running and needs its LAST sampled token
# as prev_sampled the next time it *is* scheduled. Replacing self._sampled
# with only the batch that just ran drops that entry -> KeyError in
# Executor.submit's `prev_sampled[rid]` lookup on the request's next decode.
#
# Getting two genuinely concurrent decode requests to fall out of a static
# SchedulerConfig cap isn't reproducible from the outside (once N requests are
# admitted together under a fixed max_num_seqs/max_num_batched_tokens, that
# same cap always covers their steady-state 1-token-each decode need - the
# admission itself already proved it fits). So this test drives the engine
# directly: run both requests until they are both genuinely mid-decode (past
# their prompt, real sampled tokens already threaded), then tighten
# max_num_seqs to 1 for exactly one step - the real decode-loop cap check
# (`num_seqs >= self.config.max_num_seqs: break`) then skips the second
# request that single step, exactly as it would under a tight config - and
# restore it immediately after so both requests run to completion normally.
def test_async_decode_skip_does_not_lose_prev_token():
    prompts = [[1, 2, 3], [10, 11, 12]]
    max_tokens = 6
    baseline = _llm(num_blocks=100).generate([list(p) for p in prompts], max_tokens=max_tokens)

    llm = _llm(num_blocks=100)
    ids = ["req-0", "req-1"]
    for rid, p in zip(ids, prompts):
        llm.engine.add_request(
            Request(request_id=rid, prompt_token_ids=list(p),
                    sampling=SamplingParams(max_tokens=max_tokens)))

    results: dict[str, list[int]] = {}
    squeezed = False
    while llm.engine.has_unfinished():
        both_decoding = False
        if not squeezed:
            running = {r.request_id: r for r in llm.engine.scheduler.running}
            both_decoding = all(
                rid in running and len(running[rid].output_token_ids) >= 1
                for rid in ids
            )
        if both_decoding:
            squeezed = True
            llm.engine.scheduler.config.max_num_seqs = 1
            for req in llm.engine.step():                  # must not KeyError
                results[req.request_id] = list(req.output_token_ids)
            llm.engine.scheduler.config.max_num_seqs = 256
            continue
        for req in llm.engine.step():
            results[req.request_id] = list(req.output_token_ids)

    assert squeezed, "test setup never reached a state with both requests decoding"
    tight_or_capped_out = [results[rid] for rid in ids]
    assert tight_or_capped_out == baseline       # skipped request's stream is unaffected
    assert all(len(o) == max_tokens for o in tight_or_capped_out)


def test_preempt_evicts_stale_sampled_token():
    """A preempted request is reset to recompute; its retained last-sampled
    token is invalid and MUST be dropped from engine._sampled the moment it is
    preempted. Leaving it corrupts NpuModelRunner's recompute decode-splice (a
    regression only reproducible on NPU); this guards the eviction mechanism
    host-side by driving step() directly."""
    from auto_infer.engine.request import Request, SamplingParams
    llm = _llm(num_blocks=3)                     # tight -> forces preemption
    eng = llm.engine
    for rid, p in (("a", [1, 2, 3]), ("b", [4, 5, 6])):
        eng.add_request(Request(request_id=rid, prompt_token_ids=list(p),
                                sampling=SamplingParams(max_tokens=8)))
    fired = {"n": 0}
    orig = eng.scheduler.preempt_one

    def wrapped():
        v = orig()
        if v is not None:
            fired["n"] += 1
        return v

    eng.scheduler.preempt_one = wrapped
    steps = 0
    while eng.has_unfinished() and steps < 500:
        before = fired["n"]
        eng.step()
        steps += 1
        if fired["n"] > before:
            # a preemption happened this step: no waiting (pre-recompute) request
            # may still carry a token in _sampled
            waiting_rids = {r.request_id for r in eng.scheduler.waiting}
            assert not (waiting_rids & set(eng._sampled)), (
                f"preempted/waiting rid left a stale token in _sampled: "
                f"{waiting_rids & set(eng._sampled)}")
    assert fired["n"] > 0, "preemption never fired; test did not exercise the path"


def test_recompute_discards_all_request_local_execution_state():
    from collections import deque

    llm = _llm(num_blocks=100)
    engine = llm.engine
    request = Request("r", [1, 2], SamplingParams(max_tokens=2))
    engine.add_request(request)
    request.spec_draft = [9]
    engine._sampled[request.request_id] = object()
    engine._pending_idx[request.request_id] = deque([0])

    engine._discard_recomputed_state(request.request_id)

    assert request.spec_draft == []
    assert request.request_id not in engine._sampled
    assert request.request_id not in engine._pending_idx


def test_release_finished_owns_all_engine_and_scheduler_cleanup():
    from collections import deque

    llm = _llm(num_blocks=100)
    engine = llm.engine
    request = Request("r", [1, 2], SamplingParams(max_tokens=1))
    engine.add_request(request)
    engine._sampled[request.request_id] = object()
    engine._pending_idx[request.request_id] = deque([0])

    engine._release_finished([request])

    assert request.status is RequestStatus.FINISHED
    assert engine.scheduler.get_request_or_none(request.request_id) is None
    assert request.request_id not in engine._sampled
    assert request.request_id not in engine._pending_idx


def test_async_rejects_history_dependent_sampling():
    llm = _llm(num_blocks=100, async_scheduling=True)

    with pytest.raises(
            RequestRejectedError, match="history-independent greedy"):
        llm.engine.add_request(Request(
            "r", [1, 2], SamplingParams(
                max_tokens=2, repetition_penalty=1.1)))


def test_async_rejects_executor_without_isolated_submission_slots():
    cfg = EngineConfig(
        model=ModelConfig("/mock"), async_scheduling=True)

    with pytest.raises(ValueError, match="isolated in-flight submission slots"):
        LLM(cfg, executor=Executor())


def test_async_eos_defers_kv_reclaim_until_last_submission_drains():
    llm = _llm(num_blocks=100, async_scheduling=True)
    engine = llm.engine
    engine.add_request(Request(
        "r", [1, 2, 3], SamplingParams(max_tokens=4, eos_token_id=4)))

    finished = engine.step()

    assert [request.request_id for request in finished] == ["r"]
    assert engine._queue
    assert engine.scheduler.get_request_or_none("r") is not None
    assert engine.scheduler.block_tables["r"]

    engine.step()

    assert engine.scheduler.get_request_or_none("r") is None
    assert "r" not in engine.scheduler.block_tables


def test_async_abort_defers_kv_reclaim_until_last_submission_drains():
    llm = _llm(num_blocks=100, async_scheduling=True)
    engine = llm.engine
    engine.add_request(Request(
        "r", [1, 2, 3], SamplingParams(max_tokens=4)))
    engine.step()
    assert engine._queue

    engine.abort("r")

    assert engine.scheduler.get_request_or_none("r") is not None
    assert engine.scheduler.block_tables["r"]
    while engine._queue:
        engine.step()
    assert engine.scheduler.get_request_or_none("r") is None
