import pytest

from auto_infer.config import SchedulerConfig
from auto_infer.errors import RequestRejectedError
from auto_infer.engine.kv_cache_manager import KVCacheManager
from auto_infer.engine.request import Request, SamplingParams, RequestStatus
from auto_infer.engine.scheduler import Scheduler


def make_sched(max_tokens=64, **kw):
    cfg = SchedulerConfig(max_num_batched_tokens=max_tokens, **kw)
    kv = KVCacheManager(num_blocks=100, block_size=4)
    return Scheduler(cfg, kv)


def req(rid, prompt_len, max_tokens=4):
    return Request(request_id=rid, prompt_token_ids=list(range(prompt_len)),
                   sampling=SamplingParams(max_tokens=max_tokens))


def test_prefill_whole_prompt_within_budget():
    s = make_sched(max_tokens=64)
    s.add_request(req("a", 10))
    out = s.schedule()
    assert len(out.scheduled) == 1
    sr = out.scheduled[0]
    assert sr.request_id == "a" and sr.is_prefill and sr.num_tokens_to_compute == 10
    assert out.num_batched_tokens == 10
    assert len(s.block_tables["a"]) == 3  # ceil(10/4)


def test_duplicate_request_id_is_rejected():
    s = make_sched()
    s.add_request(req("same", 1))
    with pytest.raises(RequestRejectedError, match="duplicate request id: same"):
        s.add_request(req("same", 2))


def test_chunked_prefill_splits_long_prompt():
    s = make_sched(max_tokens=8, enable_chunked_prefill=True)
    s.add_request(req("a", 20))
    out = s.schedule()
    assert out.scheduled[0].num_tokens_to_compute == 8
    assert out.num_batched_tokens == 8


def test_decode_after_prefill_one_token():
    s = make_sched(max_tokens=64)
    r = req("a", 4, max_tokens=4)
    s.add_request(r)
    s.schedule()                       # prefill all 4
    r.num_computed_tokens = 4          # engine would set this
    r.status = RequestStatus.RUNNING
    r.append_output_token(99)
    s.running.append(r)
    s.waiting = []
    out = s.schedule()                 # now a decode step
    sr = out.scheduled[0]
    assert sr.is_prefill is False
    assert sr.num_tokens_to_compute == 1


def test_max_num_seqs_limit():
    s = make_sched(max_tokens=1000, max_num_seqs=1)
    s.add_request(req("a", 4))
    s.add_request(req("b", 4))
    out = s.schedule()
    assert len(out.scheduled) == 1


def test_prefix_hit_skips_recompute():
    s = make_sched(max_tokens=64)                # block_size=4 in make_sched's KV
    # First request: prompt of 2 full blocks, run to finish so blocks register.
    r1 = req("a", 8, max_tokens=1)
    s.add_request(r1)
    s.schedule()                                 # prefill allocates 2 blocks for "a"
    r1.num_computed_tokens = 8
    r1.status = RequestStatus.RUNNING
    r1.append_output_token(99)
    s.running = [r1]; s.waiting = []
    s.free_request("a")                          # registers a's full blocks
    # Second request: identical prompt -> prefix should be matched (all but last token).
    r2 = req("b", 8, max_tokens=4)
    s.add_request(r2)
    out = s.schedule()
    sr = next(x for x in out.scheduled if x.request_id == "b")
    # 8 prompt tokens, match capped to 7 -> 1 full block (4 tokens) hit.
    assert r2.num_computed_tokens == 4
    assert sr.num_tokens_to_compute == 4         # only the uncached remainder


def test_non_chunked_prefix_hit_admits_when_remainder_fits():
    # Non-chunked scheduling with a budget smaller than the full prompt (8) but
    # >= the post-prefix-hit remainder (4). Without matching before the
    # admission gate, a request with remaining=8 > budget=4 would be rejected
    # forever, since match_prefix would never run to shrink num_computed_tokens.
    #
    # Use two schedulers sharing one KVCacheManager: a large-budget one to run
    # the first request to completion and register its blocks for reuse, then
    # a small-budget (non-chunked) one on which the starvation would show up.
    kv = KVCacheManager(num_blocks=100, block_size=4)
    cfg1 = SchedulerConfig(max_num_batched_tokens=64)
    s1 = Scheduler(cfg1, kv)
    r1 = req("a", 8, max_tokens=1)
    s1.add_request(r1)
    s1.schedule()                                 # prefill allocates 2 blocks for "a"
    r1.num_computed_tokens = 8
    r1.status = RequestStatus.RUNNING
    r1.append_output_token(99)
    s1.running = [r1]; s1.waiting = []
    s1.free_request("a")                          # registers a's full blocks

    cfg2 = SchedulerConfig(max_num_batched_tokens=4, enable_chunked_prefill=False)
    s2 = Scheduler(cfg2, kv)
    r2 = req("b", 8, max_tokens=4)
    s2.add_request(r2)
    out = s2.schedule()
    # Must be scheduled this step (not starved): matched 1 block (4 tokens),
    # remainder 4 <= budget 4 -> admitted with only the uncached remainder.
    assert len(out.scheduled) == 1
    sr = out.scheduled[0]
    assert sr.request_id == "b"
    assert r2.num_computed_tokens == 4
    assert sr.num_tokens_to_compute == 4


def test_priority_orders_waiting():
    s = make_sched(max_tokens=4, max_num_seqs=1)   # only one can prefill this step
    lo = req("lo", 4); lo.priority = 0
    hi = req("hi", 4); hi.priority = 5
    s.add_request(lo); s.add_request(hi)            # lo arrived first
    out = s.schedule()
    assert out.scheduled[0].request_id == "hi"     # higher priority wins


def test_long_prefill_token_cap():
    s = make_sched(max_tokens=64, long_prefill_token_threshold=8)
    s.add_request(req("a", 20))
    out = s.schedule()
    assert out.scheduled[0].num_tokens_to_compute == 8   # capped below budget


def test_long_prefill_cap_is_per_request_for_continuous_batching():
    s = make_sched(max_tokens=64, long_prefill_token_threshold=8)
    s.add_request(req("a", 20))
    s.add_request(req("b", 20))

    out = s.schedule()

    assert [(row.request_id, row.num_tokens_to_compute)
            for row in out.scheduled] == [("a", 8), ("b", 8)]
    assert out.num_batched_tokens == 16


def test_long_prefill_cap_does_not_stall_non_chunked_prefill():
    s = make_sched(
        max_tokens=64, enable_chunked_prefill=False,
        long_prefill_token_threshold=8)
    s.add_request(req("a", 20))

    out = s.schedule()

    assert [(row.request_id, row.num_tokens_to_compute)
            for row in out.scheduled] == [("a", 20)]


def test_needs_preemption_when_decode_cannot_grow():
    cfg = SchedulerConfig(max_num_batched_tokens=64)
    kv = KVCacheManager(num_blocks=1, block_size=4)      # 1 block total
    s = Scheduler(cfg, kv)
    r = req("a", 4, max_tokens=10)
    s.add_request(r)
    s.schedule()                                          # prefill uses the only block
    r.num_computed_tokens = 4; r.status = RequestStatus.RUNNING
    r.append_output_token(9)                              # now 5 tokens -> needs 2nd block
    s.running = [r]; s.waiting = []
    out = s.schedule()
    assert out.needs_preemption is True


def test_preempt_one_recycles_victim():
    cfg = SchedulerConfig(max_num_batched_tokens=64)
    kv = KVCacheManager(num_blocks=100, block_size=4)
    s = Scheduler(cfg, kv)
    r = req("a", 4, max_tokens=10)
    s.add_request(r); s.schedule()
    r.num_computed_tokens = 4; r.status = RequestStatus.RUNNING
    r.append_output_token(9); s.running = [r]; s.waiting = []
    assert s.preempt_one() == "a"
    assert r in s.waiting and r not in s.running
    assert r.num_computed_tokens == 0
    assert r.num_prefill_tokens == 5                      # prompt(4) + generated(1)
    assert "a" not in s.block_tables


def test_preempt_one_returns_false_when_no_running():
    s = make_sched()
    assert s.preempt_one() is None


def test_partial_match_then_alloc_fail_defers_cleanly():
    # block_size=4. Tight KVCacheManager (2 blocks) so that after "a"'s block
    # is registered+freed (becomes an evictable cached block) and the other
    # block is held by some other live allocation, "b"'s post-prefix-match
    # remainder can't be allocated -> must trigger the rollback path (not the
    # old buggy defer that left num_computed_tokens/bt partially mutated).
    kv = KVCacheManager(num_blocks=2, block_size=4)
    cfg = SchedulerConfig(max_num_batched_tokens=64)
    s = Scheduler(cfg, kv)

    r1 = req("a", 4, max_tokens=1)
    s.add_request(r1)
    s.schedule()                                 # prefill allocates a's 1 block
    r1.num_computed_tokens = 4
    r1.status = RequestStatus.RUNNING
    r1.append_output_token(99)
    s.running = [r1]; s.waiting = []
    s.free_request("a")                          # registers + frees -> a's block becomes cached

    filler = s.kv.allocate(4)                     # consume the other free block directly
    assert s.kv.num_free_blocks() == 1            # 0 free, 1 cached (a's), 1 active (filler)

    r2 = req("b", 8, max_tokens=4)                # shares first 4 tokens with "a"
    s.add_request(r2)
    out = s.schedule()

    # (a) deferred cleanly: not scheduled, block table rolled back, no progress recorded.
    assert not any(sr.request_id == "b" for sr in out.scheduled)
    assert "b" not in s.block_tables
    assert r2.num_computed_tokens == 0
    # (b) nothing else was scheduled this round -> preemption signal raised.
    assert out.needs_preemption is True

    # Free the filler allocation to make room, then retry.
    s.kv.free(filler)
    out2 = s.schedule()
    sr = next(x for x in out2.scheduled if x.request_id == "b")
    block_size = s.kv.block_size
    assert len(s.block_tables["b"]) * block_size >= r2.num_computed_tokens + sr.num_tokens_to_compute


def test_scheduled_prefill_block_table_covers_tokens():
    # Invariant guard: every scheduled prefill's block table must cover at
    # least (num_computed_tokens + num_tokens_to_compute) tokens, in a normal
    # roomy multi-request schedule.
    s = make_sched(max_tokens=64)
    s.add_request(req("a", 10))
    s.add_request(req("b", 6))
    s.add_request(req("c", 20))
    out = s.schedule()
    block_size = s.kv.block_size
    assert out.scheduled                          # sanity: something got scheduled
    for sr in out.scheduled:
        if not sr.is_prefill:
            continue
        rq = s.get_request(sr.request_id)
        assert (len(s.block_tables[sr.request_id]) * block_size
                >= rq.num_computed_tokens + sr.num_tokens_to_compute)


def test_preempt_preserves_output_and_enables_prefix_reuse():
    s = make_sched(max_tokens=64)                 # block_size=4 in make_sched's KV
    r = req("a", 8, max_tokens=4)
    s.add_request(r)
    s.schedule()                                  # prefill allocates 2 blocks for "a"
    r.num_computed_tokens = 8
    r.status = RequestStatus.RUNNING
    r.append_output_token(91)
    r.append_output_token(92)
    s.running = [r]; s.waiting = []
    assert s.preempt_one() == r.request_id
    assert r.output_token_ids == [91, 92]         # generated tokens preserved
    assert r.num_computed_tokens == 0
    assert r.num_prefill_tokens == 10             # prompt(8) + generated(2)
    # victim's own prompt prefix should now be revivable from the evictable
    # cache (registered before free), so recompute can hit it instead of
    # recomputing from scratch (spec §4a).
    matched = s.kv.match_prefix(r.prompt_token_ids[:r.num_prompt_tokens - 1])
    assert matched != []
