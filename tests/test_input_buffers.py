"""SP2: NpuModelRunner._build must switch to persistent, vectorized-fill device
buffers (see docs/superpowers/specs/2026-07-17-skeleton-sp2-persistent-input-buffers.md)
WITHOUT changing the values it produces. This test pins down today's `_build`
semantics with a faithful inline reference (`_old_build`, a literal copy of the
pre-SP2 Python-loop algorithm) and checks the real `NpuModelRunner._build`
against it for four batch shapes:
  (a) pure decode, (b) mixed prefill+decode, (c) chunked prefill,
  (d) prefix-cache-hit (num_computed_tokens > 0 at prefill start).

Runs entirely on CPU (no NPU) using a fake model/scheduler.
"""
import torch

from auto_infer.engine.request import Request, SamplingParams
from auto_infer.engine.execution import BatchPlan
from auto_infer.engine.scheduler import ScheduledRequest, SchedulerOutput
from auto_infer.worker.model_runner import NpuModelRunner

DEVICE = torch.device("cpu")


class FakeCfg:
    def __init__(self, head_dim=8, num_layers=2):
        self.head_dim = head_dim
        self.num_layers = num_layers


class FakeModel:
    """Mirrors the attrs NpuModelRunner reads from a model."""
    USES_FORWARD_CONTEXT = True

    def __init__(self):
        self.device = DEVICE
        self.dtype = torch.float32
        self.n_q_local = 2
        self.n_kv_local = 2
        self.cfg = FakeCfg()


class FakeScheduler:
    """Mirrors the subset of Scheduler's interface `_build` reads:
    `.block_tables[rid]` and `.get_request(rid)`."""

    def __init__(self):
        self.block_tables: dict[str, list[int]] = {}
        self._reqs: dict[str, Request] = {}

    def add(self, req: Request, block_table: list[int]) -> None:
        self._reqs[req.request_id] = req
        self.block_tables[req.request_id] = block_table

    def get_request(self, request_id: str) -> Request:
        return self._reqs[request_id]


def make_req(rid, prompt_len, num_output=0, num_computed_tokens=0, num_prefill_tokens=None):
    prompt = [1000 + rid * 100 + i for i in range(prompt_len)]
    req = Request(
        request_id=str(rid),
        prompt_token_ids=prompt,
        sampling=SamplingParams(max_tokens=32),
        num_prefill_tokens=num_prefill_tokens if num_prefill_tokens is not None else -1,
    )
    for i in range(num_output):
        req.append_output_token(9000 + rid * 10 + i)
    req.num_computed_tokens = num_computed_tokens
    return req


def _old_build(sched_output: SchedulerOutput, scheduler: FakeScheduler, bs: int, device):
    """Literal copy of the pre-SP2 `NpuModelRunner._build` algorithm (Python
    per-token loops + fresh torch.tensor(...)). Faithful reference: pins down
    the exact values/dtypes/lists the old code produced, independent of
    ForwardContext/attn_backend/kv_caches (irrelevant to value parity here)."""
    flat_ids, flat_pos, slots = [], [], []
    cu_q, kv_lens, bt_rows = [], [], []
    sample_idx: dict[str, int] = {}
    decode_splice: list = []
    qacc = 0
    max_blocks = 0
    all_decode = True
    for sr in sched_output.scheduled:
        req = scheduler.get_request(sr.request_id)
        n = sr.num_tokens_to_compute
        start = req.num_computed_tokens
        bt = scheduler.block_tables[sr.request_id]
        for j in range(n):
            pos = start + j
            if pos >= req.num_prompt_tokens:
                decode_splice.append((len(flat_ids), sr.request_id))
            else:
                all_decode = False
            flat_ids.append(req.all_token_ids[pos])
            flat_pos.append(pos)
            slots.append(bt[pos // bs] * bs + pos % bs)
        qacc += n
        cu_q.append(qacc)
        kv_lens.append(start + n)
        sample_idx[sr.request_id] = qacc - 1
        bt_rows.append(list(bt))
        max_blocks = max(max_blocks, len(bt))
    block_table = torch.zeros((len(bt_rows), max_blocks), dtype=torch.int32, device=device)
    for r, row in enumerate(bt_rows):
        block_table[r, :len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    return {
        "token_ids": torch.tensor(flat_ids, dtype=torch.long, device=device),
        "positions": torch.tensor(flat_pos, dtype=torch.long, device=device),
        "slot_mapping": torch.tensor(slots, dtype=torch.int32, device=device),
        "block_table": block_table,
        "cu_seqlens_q": cu_q,
        "seqlens_kv": kv_lens,
        "sample_idx": sample_idx,
        "decode_splice": decode_splice,
        "is_decode": all_decode,
    }


def _make_runner(block_size=4, num_blocks=64, **caps):
    model = FakeModel()
    from auto_infer.layers.attention.gqa import GqaFIABackend
    cfg = model.cfg
    backend = GqaFIABackend(n_q_heads=model.n_q_local, n_kv_heads=model.n_kv_local,
                            head_dim=cfg.head_dim, scale=cfg.head_dim ** -0.5,
                            num_layers=cfg.num_layers, device=model.device,
                            dtype=model.dtype)
    attention = backend, backend.alloc_kv_caches(num_blocks, block_size)
    return NpuModelRunner(model, num_blocks, block_size, attention=attention, **caps)


def _check(sched_output, scheduler, runner, block_size):
    ref = _old_build(sched_output, scheduler, block_size, DEVICE)
    ctx, sample_idx, decode_splice = runner._build(
        BatchPlan.from_scheduler(sched_output, scheduler))

    assert torch.equal(ctx.token_ids, ref["token_ids"])
    assert torch.equal(ctx.positions, ref["positions"])
    assert torch.equal(ctx.slot_mapping, ref["slot_mapping"])
    assert torch.equal(ctx.block_table, ref["block_table"])
    assert ctx.block_table.shape == ref["block_table"].shape

    assert ctx.token_ids.dtype == torch.int64 == ref["token_ids"].dtype
    assert ctx.positions.dtype == torch.int64 == ref["positions"].dtype
    assert ctx.slot_mapping.dtype == torch.int32 == ref["slot_mapping"].dtype
    assert ctx.block_table.dtype == torch.int32 == ref["block_table"].dtype

    assert ctx.cu_seqlens_q == ref["cu_seqlens_q"]
    assert ctx.seqlens_kv == ref["seqlens_kv"]
    assert ctx.is_decode == ref["is_decode"]
    assert sample_idx == ref["sample_idx"]
    assert decode_splice == ref["decode_splice"]


def test_pure_decode():
    bs = 4
    scheduler = FakeScheduler()
    sched = []
    for rid in range(3):
        req = make_req(rid, prompt_len=5, num_output=1, num_computed_tokens=5)
        scheduler.add(req, [10 + rid * 2, 11 + rid * 2])   # 2 blocks: covers pos<8
        sched.append(ScheduledRequest(str(rid), 1, False, list(scheduler.block_tables[str(rid)])))
    so = SchedulerOutput(sched, num_batched_tokens=3)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    _check(so, scheduler, runner, bs)


def test_mixed_prefill_decode():
    bs = 4
    scheduler = FakeScheduler()
    # decode request (already prefilled)
    dreq = make_req(0, prompt_len=5, num_output=1, num_computed_tokens=5)
    scheduler.add(dreq, [10, 11])
    # fresh prefill request, full (non-chunked) prompt
    preq = make_req(1, prompt_len=4, num_output=0, num_computed_tokens=0)
    scheduler.add(preq, [20])                              # 1 block: covers pos<4
    sched = [
        ScheduledRequest("0", 1, False, list(scheduler.block_tables["0"])),
        ScheduledRequest("1", 4, True, list(scheduler.block_tables["1"])),
    ]
    so = SchedulerOutput(sched, num_batched_tokens=5)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    _check(so, scheduler, runner, bs)


def test_chunked_prefill():
    bs = 4
    scheduler = FakeScheduler()
    # prompt len 10, first chunk of 4 already computed, this step computes 3 more (partial)
    req = make_req(0, prompt_len=10, num_output=0, num_computed_tokens=4)
    scheduler.add(req, [30, 31, 32])                        # 3 blocks: covers pos<12
    sched = [ScheduledRequest("0", 3, True, list(scheduler.block_tables["0"]))]
    so = SchedulerOutput(sched, num_batched_tokens=3)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    _check(so, scheduler, runner, bs)


def test_prefix_cache_hit():
    bs = 4
    scheduler = FakeScheduler()
    # prompt len 8, prefix-cache match already covered first 5 tokens (num_computed_tokens=5
    # set directly, mimicking Scheduler.schedule()'s prefix-match path), this step computes
    # the remaining 3 tokens to finish the prompt.
    req = make_req(0, prompt_len=8, num_output=0, num_computed_tokens=5)
    scheduler.add(req, [40, 41])                            # 2 blocks: covers pos<8
    sched = [ScheduledRequest("0", 3, True, list(scheduler.block_tables["0"]))]
    so = SchedulerOutput(sched, num_batched_tokens=3)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    _check(so, scheduler, runner, bs)


def test_ctx_slices_are_views_of_persistent_buffers():
    """ctx.token_ids must be a mutable VIEW into the runner's persistent buffer
    (required for submit()'s decode-splice write `tok[fidx] = prev_sampled[rid]`),
    not a fresh tensor each call."""
    bs = 4
    scheduler = FakeScheduler()
    req = make_req(0, prompt_len=5, num_output=1, num_computed_tokens=5)
    scheduler.add(req, [10, 11])
    sched = [ScheduledRequest("0", 1, False, list(scheduler.block_tables["0"]))]
    so = SchedulerOutput(sched, num_batched_tokens=1)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    ctx, _, _ = runner._build(BatchPlan.from_scheduler(so, scheduler))
    assert ctx.token_ids.data_ptr() == runner.token_ids_buf.data_ptr()
    ctx.token_ids[0] = 424242
    assert runner.token_ids_buf[0].item() == 424242


def test_spec_build_accepts_immutable_multi_token_drafts():
    scheduler = FakeScheduler()
    request = make_req(0, prompt_len=3, num_output=1,
                       num_computed_tokens=3)
    request.spec_draft = (7, 8, 9)
    scheduler.add(request, [10, 11])
    output = SchedulerOutput(
        [ScheduledRequest("0", 4, False, [10, 11])],
        num_batched_tokens=4)
    runner = _make_runner(
        block_size=4, max_num_batched_tokens=8, max_num_seqs=2,
        max_model_len=8)

    ctx, _, _ = runner._build(BatchPlan.from_scheduler(output, scheduler))

    assert ctx.token_ids.tolist() == [9000, 7, 8, 9]


def test_second_step_no_stale_carryover():
    """A later, SHORTER step must not see leftover values from an earlier,
    LONGER step in the persistent buffers (used-prefix is always freshly filled)."""
    bs = 4
    scheduler = FakeScheduler()
    big = make_req(0, prompt_len=8, num_output=0, num_computed_tokens=0)
    scheduler.add(big, [50, 51])
    so_big = SchedulerOutput(
        [ScheduledRequest("0", 8, True, list(scheduler.block_tables["0"]))], num_batched_tokens=8)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=32, max_num_seqs=8, max_model_len=32)
    runner._build(BatchPlan.from_scheduler(so_big, scheduler))

    scheduler2 = FakeScheduler()
    small = make_req(1, prompt_len=5, num_output=1, num_computed_tokens=5)
    scheduler2.add(small, [60, 61])
    so_small = SchedulerOutput(
        [ScheduledRequest("1", 1, False, list(scheduler2.block_tables["1"]))], num_batched_tokens=1)
    ctx, _, _ = runner._build(BatchPlan.from_scheduler(so_small, scheduler2))
    ref = _old_build(so_small, scheduler2, bs, DEVICE)
    assert torch.equal(ctx.token_ids, ref["token_ids"])
    assert ctx.token_ids.shape == (1,)


def test_auto_grow_beyond_initial_caps():
    """A step larger than the ctor caps must AUTO-GROW the buffers (no numpy
    broadcast crash — the SP2 review's Critical) and still produce values
    identical to the exact-sized reference. Tiny caps force growth on tokens,
    num_seqs, AND block_table cols simultaneously."""
    bs = 4
    scheduler = FakeScheduler()
    a = make_req(0, prompt_len=20, num_output=0, num_computed_tokens=0)   # 20 tok, 5 blocks
    scheduler.add(a, list(range(5)))
    b = make_req(1, prompt_len=8, num_output=0, num_computed_tokens=0)    # forces num_reqs=2
    scheduler.add(b, [20, 21])
    sched = [ScheduledRequest("0", 20, True, list(range(5))),
             ScheduledRequest("1", 8, True, [20, 21])]
    so = SchedulerOutput(sched, num_batched_tokens=28)
    runner = _make_runner(block_size=bs, max_num_batched_tokens=4,
                          max_num_seqs=1, max_model_len=8)               # tiny caps
    _check(so, scheduler, runner, bs)                                    # no crash + value-identical
    assert runner._buf_tokens >= 28 and runner._buf_seqs >= 2 and runner._buf_blocks >= 5
