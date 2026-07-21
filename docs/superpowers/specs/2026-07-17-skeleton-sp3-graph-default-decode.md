# SP3 — Graph-default decode via GraphAttentionBackend (skeleton refactor 3/5)

Status: Draft · Date: 2026-07-17 · Branch: feat/skeleton-sp3 · Target: npu2

## Context & goal
The benchmark showed decode is host-bound (~30 ms/token, 0.5B; device ~1 ms) — dominated
by the 24-layer eager Python dispatch per step. ACL-graph capture/replay eliminates that
(one graph replay ≈ µs of host dispatch). `worker/graph_decode_runner.py` ALREADY does
this and works (`verify_qwen2_graphdecode_batched` passes) — but it is a SEPARATE 4th
forward (`Qwen2GraphAdapter`), sync-only, and non-default.

**SP3 rebuilds graph decode as a `GraphAttentionBackend` (a new `AttentionBackend`), so the
UNIFIED `forward(ctx)` is what gets graph-captured** — graph vs eager becomes just the
injected backend, not a reimplemented forward. Deletes `Qwen2GraphAdapter`. Makes graph
the DEFAULT decode path (eager prefill/mixed unchanged). This is the big decode-latency win.

**All the ops here are NPU-only** (`npu_scatter_pa_kv_cache`, FIA-v2 `.out`,
`graph_task_group_*`, `NPUGraph`) — NOT host-testable. The implementer writes it by
REFACTORING the working `graph_decode_runner.py` (proven reference); correctness is gated
entirely on npu2 runs the controller drives.

## Design
Reuse the proven capture/replay machinery; change only WHERE the per-layer compute comes
from (unified `forward(ctx)` instead of `Qwen2GraphAdapter.forward`).

### `layers/attention/backend.py` — add `GraphAttentionBackend(AttentionBackend)`
- `alloc_kv_caches(num_blocks, block_size)`: NZ-layout KV per layer (as the adapter's
  `alloc_caches`), using ctor `num_layers/device/dtype` (ABC-uniform, like PagedFIABackend).
- `write_kv(layer_idx, k, v, ctx)`: `npu_scatter_pa_kv_cache` with NZ views (adapter `_store`).
- `attn(layer_idx, q, k, v, ctx)`: FIA-v2 `.out` variant. Two sub-modes on the backend:
  - **capture** (`self.capturing=True`): wrap in `graph_task_group_begin/end`, append the
    `(handle, q, knz, vnz, o, lse)` to `self.reg[layer_idx]`; return `o`.
  - **replay/eager**: plain `.out` call; return `o`.
- `update(ctx)`: for each layer's registered handle, `graph_task_update_*` with the step's
  `actual_seq_kvlen` (adapter `update`). Called by the runner before `graph.replay()`.
- Carries NZ constant, scale, head dims (ctor). The unified `forward(ctx)` is unchanged —
  it just calls `be.write_kv`/`be.attn`; the graph wrapping happens inside the backend.

### `worker/graph_decode_runner.py` — rebuild `GraphPagedRunner` on the unified forward
- Keep gears `[1,2,4,8,16,32,64]` capped by `max_gear`, `_Gear` static buffers, scratch
  blocks, `stats`.
- `_capture(g)`: build a decode `ForwardContext` over the gear's static buffers with a
  `GraphAttentionBackend(capturing=True)`; warmup `model.forward(ctx)`; then
  `with torch.npu.graph(graph): model.forward(ctx)` (backend registers per-layer handles).
- `_graph(sched, scheduler, gear)`: marshal host→static buffers (reuse SP2-style bulk fill),
  `backend.update(ctx)`, `gear.graph.replay()`, `model.logits(hout[:B])`, batched sample.
- `_eager(...)`: decode/prefill/mixed/oversized fallback = `model.forward(ctx)` with a
  `PagedFIABackend` (the SP1/SP2 path) — NOT a reimplemented forward.
- DELETE `Qwen2GraphAdapter` entirely (its compute now lives in the unified forward).
- `execute()`: decode-only & B≤max_gear → `_graph`; else `_eager`. (Batched sampling from SP1.)

### Default wiring
- Make graph decode the DEFAULT for Qwen2: `GraphPagedNpuExecutor` gains `force_eager` (exists)
  and becomes the default executor the engine builds for a graph-capable model, OR a config
  flag `EngineConfig.graph_decode=True`. Keep eager path selectable. (Engine/scheduler stay
  otherwise untouched; only executor selection changes.)
- SP3 keeps the graph executor SYNC (`execute`); composing graph replay with the async
  batch-queue + worker/output threads is SP4.

## Verification (npu2, Qwen2.5-0.5B) — NPU-only
- **Graph == eager parity**: `verify_qwen2_graphdecode_batched` (existing) must still PASS
  with the rebuilt runner (graph output argmax == eager reference).
- New `tools/parity_graph.py` (or extend parity.py): unified `forward(ctx)`+GraphBackend
  (via a 1-gear capture/replay) vs +PagedFIABackend, argmax match.
- Full regression: parity, smoke_engine_npu, prefix_cache, preemption, host suite.
- **Bench**: graph-default decode vs SP2 eager decode (`bench_async_scheduling` adapted, or a
  new decode-latency micro-bench) — EXPECT a large decode tok/s gain (the 30 ms→µs dispatch
  win). Report the delta.
- Host-testable slice: gear selection (`_get_gear`), host-side buffer marshaling logic
  (pure Python) — unit-test what doesn't need NPU ops.

## Files
- Modify: `layers/attention/backend.py` (+GraphAttentionBackend), `worker/graph_decode_runner.py`
  (rebuild on forward(ctx), delete Qwen2GraphAdapter), executor selection (engine/executor
  wiring or config), maybe `tools/parity_graph.py`.
- Untouched: engine step loop, scheduler, KV manager, models (forward(ctx) already serves),
  PagedFIABackend/DenseBackend, deepseek.

## Risks
- ACL-graph capture requires the unified `forward(ctx)` to be capture-clean (no host branches
  on device values, static shapes). Verify: `_rope_cos_sin`, `_add_rms_norm`, `_swiglu`,
  `_lin_param` are all capturable (they were, inside the adapter). The rope table recompute
  per forward is captured as constants for a fixed gear — fine.
- NZ layout differs from PagedFIABackend's layout → GraphAttentionBackend owns its own cache;
  the eager fallback uses a DIFFERENT (PagedFIABackend) cache. Two caches coexist (memory),
  OR the runner uses one NZ cache for both (eager FIA-v2 also reads NZ — the current
  `_eager` already does). Prefer: single NZ cache, both graph and eager read it (matches
  today's graph_decode_runner which shares `self.caches`).
