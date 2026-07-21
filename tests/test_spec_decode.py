"""MTP speculative decoding (EngineCore routing + scheduler per-request qd),
host-side with a deterministic fake MTP executor (no NPU)."""
import pytest

from auto_infer.config import (CacheConfig, EngineConfig, ModelConfig, SchedulerConfig,
                                SpecDecodeConfig)
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.execution import ExecutionResult, ExecutionStats
from auto_infer.entrypoints.llm import LLM
from auto_infer.spec_decode.geometry import MtpGeometry
from auto_infer.spec_decode.layout import confirmed_layout
from auto_infer.worker.graph_mtp_runner import (
    _DrafterGear, _TargetGear, _chunk_spec_requests,
    _decode_packed_results, _next_mtp_tokens,
    _finishes_after_one, _reachable_drafter_pairs, _select_drafter_gear)


def _plain(prompts, max_tokens, num_blocks=200):
    cfg = EngineConfig(model=ModelConfig(model_path="/mock"),
                       cache=CacheConfig(block_size=4, num_blocks=num_blocks),
                       scheduler=SchedulerConfig(max_num_batched_tokens=256))
    return LLM(cfg, executor=MockExecutor(vocab_size=1000)).generate(
        [list(p) for p in prompts], max_tokens=max_tokens)


class _FakeMtpExecutor:
    """Deterministic (last+1)-rule stand-in for a runner with an MTP head: emits
    the target's argmax, drafts the TRUE next token (always accepted) so each spec
    step confirms 2 tokens. Exercises EngineCore's MTP routing + draft carry across
    steps + scheduler per-request qd, host-side (no NPU)."""

    def __init__(self, vocab=1000):
        self.vocab = vocab

    def execute_spec_mtp(self, plan):
        emitted, next_drafts = {}, {}
        acc = steps = 0
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            n = sr.num_tokens_to_compute
            base = req.num_computed_tokens
            if base + n < req.num_prefill_tokens:
                continue
            if sr.is_prefill or not req.spec_draft:
                source_row = (req.num_prefill_tokens - 1
                              if sr.is_prefill else base)
                tok0 = (req.all_token_ids[source_row] + 1) % self.vocab
                emitted[sr.request_id] = [tok0]
                next_drafts[sr.request_id] = [(tok0 + 1) % self.vocab]     # perfect next draft
            else:
                q0, d = req.all_token_ids[base], req.spec_draft[0]
                p0, p1 = (q0 + 1) % self.vocab, (d + 1) % self.vocab
                steps += 1
                if d == p0:                                                # accept (perfect draft)
                    emitted[sr.request_id] = [p0, p1]
                    next_drafts[sr.request_id] = [(p1 + 1) % self.vocab]
                    acc += 1
                else:
                    emitted[sr.request_id] = [p0]
                    next_drafts[sr.request_id] = [(p0 + 1) % self.vocab]
        return ExecutionResult(
            tokens={rid: tuple(tokens) for rid, tokens in emitted.items()},
            next_drafts={rid: tuple(tokens) for rid, tokens in next_drafts.items()},
            stats=ExecutionStats(accepted=acc, steps=steps))


def _mtp_llm():
    cfg = EngineConfig(model=ModelConfig(model_path="/mock"),
                       cache=CacheConfig(block_size=4, num_blocks=200),
                       scheduler=SchedulerConfig(max_num_batched_tokens=256),
                       spec_decode=SpecDecodeConfig())
    return LLM(cfg, executor=_FakeMtpExecutor())


def test_mtp_routing_matches_plain_greedy():
    prompts = [[1, 2, 3], [10, 11]]
    plain = _plain(prompts, 8)
    spec = _mtp_llm().generate([list(p) for p in prompts], max_tokens=8)
    assert spec == plain                                   # 2 tok/step, same greedy stream
    assert all(len(o) == 8 for o in spec)


def test_mtp_exact_length_cap():
    spec = _mtp_llm().generate([[1, 2, 3]], max_tokens=7)   # odd cap: last step emits 1 of 2
    assert len(spec[0]) == 7


@pytest.mark.parametrize("sampling", [
    {"temperature": 0.7},
    {"top_k": 5},
    {"top_p": 0.9},
    {"presence_penalty": 0.1},
    {"logit_bias": {1: 2.0}},
    {"allowed_token_ids": [1, 2]},
])
def test_mtp_rejects_sampling_semantics_it_cannot_preserve(sampling):
    from auto_infer.errors import RequestRejectedError
    from auto_infer.engine.request import Request, SamplingParams

    engine = _mtp_llm().engine
    request = Request("r", [1, 2], SamplingParams(max_tokens=2, **sampling))
    with pytest.raises(RequestRejectedError, match="greedy-only"):
        engine.add_request(request)


def test_missing_replacement_draft_clears_carried_state():
    class DropsDraftAfterVerify(_FakeMtpExecutor):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute_spec_mtp(self, plan):
            self.calls += 1
            result = super().execute_spec_mtp(plan)
            if self.calls >= 2:
                return ExecutionResult(tokens=result.tokens, stats=result.stats)
            return result

    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=200),
        scheduler=SchedulerConfig(max_num_batched_tokens=256),
        spec_decode=SpecDecodeConfig())
    engine = LLM(cfg, executor=DropsDraftAfterVerify()).engine
    request = __import__("auto_infer.engine.request", fromlist=["Request"]).Request(
        "r", [1, 2],
        __import__("auto_infer.engine.request", fromlist=["SamplingParams"]).SamplingParams(
            max_tokens=6))
    engine.add_request(request)
    engine.step()
    assert request.spec_draft

    engine.step()

    assert request.spec_draft == []


def test_mtp_final_single_token_makes_progress_at_block_boundary():
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=1),
        scheduler=SchedulerConfig(max_num_batched_tokens=16),
        spec_decode=SpecDecodeConfig())
    llm = LLM(cfg, executor=_FakeMtpExecutor())

    output = llm.generate([[1, 2, 3]], max_tokens=2)

    assert output == [[4, 5]]
    assert llm.engine.scheduler.num_preemptions == 0


def test_multistep_mtp_one_token_request_needs_no_continuation_capacity():
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=1, num_blocks=1),
        scheduler=SchedulerConfig(max_num_batched_tokens=4),
        spec_decode=SpecDecodeConfig(num_speculative_tokens=2))
    llm = LLM(cfg, executor=_FakeMtpExecutor())

    output = llm.generate([[1]], max_tokens=1)

    assert output == [[2]]
    assert llm.engine.scheduler.num_preemptions == 0
    assert len(llm.engine.kv.match_prefix([1, 2])) == 1


def test_mtp_stats_logged():
    from auto_infer.engine.metrics import StatLogger
    llm = _mtp_llm()
    llm.config.log_stats = True
    llm.engine.stat_logger = StatLogger(interval_s=1e9)
    llm.generate([[1, 2, 3]], max_tokens=8)
    sl = llm.engine.stat_logger
    assert sl._spec_steps > 0 and sl._spec_accepted > 0    # acceptance recorded


def test_scheduler_reserves_recurrent_mtp_continuation_slots():
    from auto_infer.engine.kv_cache_manager import KVCacheManager
    from auto_infer.engine.request import Request, RequestStatus, SamplingParams
    from auto_infer.engine.scheduler import Scheduler

    kv = KVCacheManager(100, 2)
    scheduler = Scheduler(
        SchedulerConfig(max_num_batched_tokens=4), kv,
        num_speculative_tokens=3)
    request = Request("a", [1, 2, 3], SamplingParams(max_tokens=20))
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 3
    request.output_token_ids = [4]
    request.spec_draft = (5, 6, 7)
    scheduler._requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.block_tables[request.request_id] = kv.allocate(4)

    output = scheduler.schedule()

    assert output.scheduled[0].num_tokens_to_compute == 4
    assert len(scheduler.block_tables[request.request_id]) * kv.block_size >= 6


def test_scheduler_drops_full_draft_near_length_cap():
    from auto_infer.engine.kv_cache_manager import KVCacheManager
    from auto_infer.engine.request import Request, RequestStatus, SamplingParams
    from auto_infer.engine.scheduler import Scheduler

    kv = KVCacheManager(4, 1)
    scheduler = Scheduler(
        SchedulerConfig(max_num_batched_tokens=4), kv,
        num_speculative_tokens=2)
    request = Request("a", [1], SamplingParams(max_tokens=3), [2])
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 1
    request.spec_draft = (3, 4)
    scheduler._requests[request.request_id] = request
    scheduler.running.append(request)
    scheduler.block_tables[request.request_id] = kv.allocate(2)

    output = scheduler.schedule()

    assert request.spec_draft == []
    assert output.scheduled[0].num_tokens_to_compute == 1


def test_mtp_prefill_next_tokens_accept_immutable_request_views():
    tokens = (10, 11, 12, 13, 14)

    assert _next_mtp_tokens(tokens, base=0, count=3,
                            complete=False, first_token=None) == [11, 12, 13]
    assert _next_mtp_tokens(tokens, base=2, count=3,
                            complete=True, first_token=99) == [13, 14, 99]


def test_final_token_detection_honors_length_and_stops():
    from auto_infer.engine.request import Request, SamplingParams
    by_length = Request("length", [1], SamplingParams(max_tokens=2), [7])
    by_eos = Request(
        "eos", [1], SamplingParams(max_tokens=5, eos_token_id=9), [7])
    blocked_eos = Request(
        "blocked", [1], SamplingParams(
            max_tokens=5, min_tokens=3, eos_token_id=9), [7])

    assert _finishes_after_one(by_length, 8)
    assert _finishes_after_one(by_eos, 9)
    assert not _finishes_after_one(blocked_eos, 9)


def test_oversized_spec_batches_split_on_largest_available_gear():
    rows = list(range(17))

    assert list(map(len, _chunk_spec_requests(rows, 16))) == [16, 1]
    assert list(map(len, _chunk_spec_requests(rows[:7], 3))) == [2, 2, 2, 1]


def test_confirmed_layout_compacts_mixed_acceptance_in_request_order():
    layout = confirmed_layout([0, 1, 0, 1], MtpGeometry(1))

    assert layout.source_rows == (0, 2, 3, 4, 6, 7)
    assert layout.query_lengths == (1, 2, 1, 2)
    assert layout.cumulative_query_lengths == (1, 3, 4, 6)
    assert layout.final_rows == (0, 2, 3, 5)
    assert layout.active_tokens == 6


def test_confirmed_layout_handles_uniform_acceptance():
    geometry = MtpGeometry(1)
    rejected = confirmed_layout([0, 0, 0], geometry)
    accepted = confirmed_layout([1, 1, 1], geometry)

    assert rejected.source_rows == (0, 2, 4)
    assert rejected.query_lengths == (1, 1, 1)
    assert rejected.cumulative_query_lengths == (1, 2, 3)
    assert rejected.final_rows == (0, 1, 2)
    assert rejected.active_tokens == 3
    assert accepted.source_rows == (0, 1, 2, 3, 4, 5)
    assert accepted.query_lengths == (2, 2, 2)
    assert accepted.cumulative_query_lengths == (2, 4, 6)
    assert accepted.final_rows == (1, 3, 5)
    assert accepted.active_tokens == 6


def test_confirmed_layout_rejects_acceptance_beyond_model_depth():
    import pytest

    with pytest.raises(ValueError, match="between 0 and 1"):
        confirmed_layout([0, 2], MtpGeometry(1))


def test_mtp_geometry_is_derived_from_contiguous_model_layers():
    geometry = MtpGeometry.from_weights({
        "model.mtp_layers.0.input_proj.weight": object(),
        "model.mtp_layers.1.input_proj.weight": object(),
    })

    assert geometry.draft_depth == 2
    assert geometry.query_width == 3
    assert geometry.layer_prefix(1) == "model.mtp_layers.1."


def test_mtp_geometry_rejects_missing_and_noncontiguous_layers():
    import pytest

    with pytest.raises(ValueError, match="no MTP layers"):
        MtpGeometry.from_weights({})
    with pytest.raises(ValueError, match="contiguous"):
        MtpGeometry.from_weights({
            "model.mtp_layers.0.input_proj.weight": object(),
            "model.mtp_layers.2.input_proj.weight": object(),
        })


def test_two_stage_drafter_gear_uses_confirmed_tokens_and_request_gear():
    geometry = MtpGeometry(1)
    assert _select_drafter_gear(6, 4, 16, geometry) == (8, 4)
    assert _select_drafter_gear(29, 16, 16, geometry) == (32, 16)
    assert _select_drafter_gear(33, 16, 16, geometry) is None
    assert _select_drafter_gear(4, 5, 4, geometry) is None
    assert _select_drafter_gear(9, 4, 16, MtpGeometry(2)) == (12, 4)


def test_two_stage_reachable_drafter_pairs_are_bounded():
    geometry = MtpGeometry(1)
    assert _reachable_drafter_pairs(4, geometry) == (
        (1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4))
    assert _reachable_drafter_pairs(16, geometry)[-2:] == ((16, 16), (32, 16))


def test_two_stage_prewarm_is_startup_only_and_fails_with_exact_gear():
    from auto_infer.worker.graph_mtp_runner import GraphMtpPagedRunner
    runner = GraphMtpPagedRunner.__new__(GraphMtpPagedRunner)
    runner.max_gear = 4
    runner.geometry = MtpGeometry(1)
    runner.target_gears = {}
    runner.drafter_gears = {}
    runner.stats = {
        "target_capture_attempts": 0, "drafter_capture_attempts": 0}
    target_calls, drafter_calls = [], []

    def capture_target(gear):
        target_calls.append(gear)
        return f"target-{gear}"

    def capture_drafter(key):
        drafter_calls.append(key)
        if key == (4, 2):
            raise RuntimeError("unsupported pair")
        return f"draft-{key}"

    runner._capture_target = capture_target
    runner._capture_drafter = capture_drafter
    with pytest.raises(RuntimeError, match=r"drafter graph capture failed for \(4, 2\)"):
        runner._prewarm_two_stage_gears()

    assert target_calls == [1, 2, 4]
    assert drafter_calls[-1] == (4, 2)


def test_two_stage_gears_own_fixed_target_and_compacted_buffers():
    import torch
    geometry = MtpGeometry(1)
    target = _TargetGear(
        4, max_blocks=8, hidden=16, device=torch.device("cpu"),
        dtype=torch.bfloat16, geometry=geometry)
    drafter = _DrafterGear(
        (8, 4), target, max_blocks=8, device=torch.device("cpu"))

    assert target.tid.shape == (8,)
    assert target.active_mask.shape == (4,)
    assert target.ep_active_token_mask.shape == (8,)
    assert target.ep_active_token_mask.dtype == torch.bool
    assert target.compact_hidden.shape == (8, 16)
    assert target.compact_tokens.shape == (8,)
    assert target.compact_positions.shape == (8,)
    assert target.compact_slots.shape == (8,)
    assert target.p_buf.shape == (4, 2)
    assert target.na_buf.shape == (4,)
    assert drafter.block_table.shape == (5, 8)
    assert drafter.sample_rows.shape == (4,)
    assert drafter.draft_buf.shape == (4, 1)
    assert drafter.target is target


def test_packed_results_decode_accept_and_reject_without_padding():
    emitted, drafts, accepted = _decode_packed_results(
        [[10, 11, 0, 20], [30, 31, 1, 40], [0, 0, 0, 0]],
        ["reject", "accept"], MtpGeometry(1))

    assert emitted == {"reject": [10], "accept": [30, 31]}
    assert drafts == {"reject": [20], "accept": [40]}
    assert accepted == 1


def test_scheduler_per_request_qd():
    """Decode query length = 1 + len(spec_draft): plain decode reserves 1, a
    1-draft spec request reserves 2 (verified via ScheduledRequest.num_tokens)."""
    from auto_infer.engine.kv_cache_manager import KVCacheManager
    from auto_infer.engine.request import Request, RequestStatus, SamplingParams
    from auto_infer.engine.scheduler import Scheduler
    sch = Scheduler(SchedulerConfig(enable_prefix_caching=False), KVCacheManager(100, 4))
    r = Request(request_id="a", prompt_token_ids=[1, 2, 3], sampling=SamplingParams(max_tokens=20))
    r.status = RequestStatus.RUNNING
    r.num_computed_tokens = 3
    r.output_token_ids = [4]                               # decode phase (num_computed = num_tokens-1)
    sch._requests["a"] = r
    sch.running.append(r)
    sch.block_tables["a"] = sch.kv.allocate(4)
    assert sch.schedule().scheduled[0].num_tokens_to_compute == 1   # no draft -> 1 query token
    r.spec_draft = [99]
    assert sch.schedule().scheduled[0].num_tokens_to_compute == 2   # 1 draft -> 2 query tokens
