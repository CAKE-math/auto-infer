# SP2 — Persistent / incremental input buffers (skeleton refactor 2/5)

Status: Draft · Date: 2026-07-17 · Branch: feat/skeleton-sp2 · Target: npu2

## Context & goal
The async benchmark proved decode is **host-bound** (~30 ms/token on 0.5B, device ~1 ms).
Part of that host cost is `NpuModelRunner._build`: every step it rebuilds
`token_ids/positions/slot_mapping/block_table` with per-request/per-token **Python loops**
and fresh `torch.tensor(...)` + H2D. SP2 replaces that with **persistent device buffers
updated incrementally / vectorized**, so per-step host marshaling is O(changed) not
O(rebuild-from-scratch). This also provides the **static input buffers** SP3's graph
capture needs (a graph replays against fixed buffer addresses).

Note: SP2 alone won't move the benchmark much (the 24-layer eager Python *forward
dispatch* still dominates — that's SP3's target). SP2's value is (a) trimming `_build`,
and (b) being the prerequisite for clean SP3 graph capture. Keep it lean.

## Scope
IN — `auto_infer/worker/model_runner.py` only:
- `NpuModelRunner` owns persistent buffers sized to caps: `token_ids`, `positions`,
  `slot_mapping` (len `max_num_batched_tokens`), `block_table`
  (`max_num_seqs × max_blocks`), plus pinned host staging (numpy) for a single bulk H2D.
- `_build` fills those buffers **vectorized (numpy)** instead of Python-append loops, and
  returns a `ForwardContext` whose fields are **slices** of the persistent buffers
  (`token_ids_buf[:T]`, `block_table_buf[:num_reqs, :max_blk]`).
- Decode fast path: B rows × 1 token — fill the first B slots; block-table rows copied
  from the scheduler's block lists (still needed since block ids can change), but via one
  batched numpy assignment + single H2D, not per-row device writes.
- Preserve EXACT semantics: the built tensors must be identical values to today's `_build`
  (dtypes int64/int32 as before; `cu_seqlens_q`/`seqlens_kv` lists unchanged; decode-splice
  index list unchanged). Behavior/parity must stay bitwise (the forward is unchanged).

OUT: engine/scheduler/model/backend/graph_decode_runner — untouched. No graph work (SP3).
`GraphPagedNpuExecutor` path unaffected. DeepSeek path (`forward_paged` via ctx fields)
must still get correct tensors from the new `_build`.

## Key constraints
- The persistent `token_ids` buffer must still allow the async decode-splice write
  (`tok[fidx] = prev_sampled[rid]`) — i.e. `ctx.token_ids` is a mutable slice/view of the
  buffer, and each step overwrites the used prefix so stale values never leak.
- Caps come from config: `max_num_batched_tokens` (scheduler), `max_num_seqs`, and
  `max_blocks = ceil(max_model_len / block_size)`. Allocate buffers once in `__init__`.
- Correctness over micro-opt: a bulk numpy-build + single H2D per buffer per step is the
  target; true in-place incremental (only touch changed decode slots) is a bonus, only if
  it stays obviously correct.

## Verification (npu2, Qwen2.5-0.5B)
- Host: a unit test that the new `_build` produces tensors equal (value-wise) to a
  reference built the old way, for (a) a pure-decode batch, (b) a mixed prefill+decode
  batch, (c) a chunked-prefill batch, (d) a prefix-cache-hit request (num_computed>0).
- NPU regression: parity, smoke_engine_npu, smoke_qwen2, prefix_cache, preemption,
  graphdecode, host suite — all still PASS (SP2 is behavior-preserving).
- `bench_async_scheduling.py`: decode tok/s should be **≥** SP1 (expect a small gain from
  cheaper `_build`; the big gain waits for SP3). Report the delta honestly.

## Files
- Modify: `auto_infer/worker/model_runner.py`
- Test: `tests/test_input_buffers.py` (new)

## Task order (TDD)
1. Reference-capture test: pin down today's `_build` output for the 4 batch shapes above
   (build a small fake Scheduler/Request set on CPU — MockExecutor-style — no NPU).
2. Add persistent buffers to `__init__` (sized from caps) + pinned numpy staging.
3. Rewrite `_build` to vectorized numpy fill + single H2D per buffer; return ctx slices.
4. Make the reference test assert new == old values for all 4 shapes.
5. NPU regression + bench delta (controller, npu2).
