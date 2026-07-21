# SP6 — MLA graph decode for DeepSeek (GraphMlaBackend)

Status: Draft · Date: 2026-07-17 · Branch: feat/mla-graph-decode · Target: npu2
Verify model: /data1/models/DeepSeek-V2-Lite-Chat

## Goal
DeepSeek-V2-Lite decodes at 15.6 tok/s (eager, host-bound) vs Qwen2's 115 tok/s (graph).
DeepSeek has no graph backend — SP3's graph was GQA-only. SP6 adds a `GraphMlaBackend` so
DeepSeek's unified `forward(ctx)` (MLA + MoE) is ACL-graph-captured for decode, like Qwen2.
Expected: a large decode speedup (the ~3.5× graph win, minus MoE overhead).

## Feasibility (confirmed by inspection + vllm-ascend)
- MLA attention graph: vllm-ascend does QS=1 MLA graph decode (`attention/mla_v1.py`) with a
  fixed **graph pad size** + FIA `.out` + per-layer graph-param update — same shape as our
  `GraphGqaBackend`. The block_size-128 FIA constraint we hit was QS>1 (prefill); decode is
  QS=1, so it should not apply.
- MoE in graph: with the batch **padded to a fixed gear size**, the MoE ops
  (`npu_moe_init_routing`/`npu_grouped_matmul`/`npu_moe_finalize_routing`) have STATIC shapes;
  the dynamic per-expert distribution rides in the **`group_list` device tensor** (data, not
  shape) which a captured op reads at replay. Our `_moe_compute_fused` is graph-clean (no host
  branching on device values — verified).

## Design
### `layers/attention/backend.py` — `GraphMlaBackend(AttentionBackend)`
Mirror `GraphGqaBackend`, but the attention body is MLA (from `MlaFIABackend`/`mla.py`):
- `alloc_kv_caches`: the MLA KV layout `MlaFIABackend` uses (non-absorbed: K=(nope+rope) per
  head, V=vd per head), in whatever layout the graph FIA `.out` needs (NZ if required, mirror
  GraphGqaBackend's NZ handling).
- `attention(layer_idx, x, lp, ctx)`: the MLA sub-block (q/kv LoRA proj, nope/rope split, the
  **DeepSeek interleaved-rope reshape + rope** — MUST keep the SP-rope-fix `_ds_rope_interleave`,
  kv write, attention, o_proj). The attention op is FIA-v2 `.out`, capture-aware: `capturing`
  flag → wrap in `graph_task_group_begin/end`, append handle+tensors to `self.reg`; else plain.
- `update(ctx)`: per registered layer handle, `graph_task_update_*` with the step's kvlens
  (like GraphGqaBackend). `begin_capture`/`end_capture`.
- Keep absorbed MLA OUT of the graph path for now (block_size-128 constraint); non-absorbed only.

### `worker/graph_decode_runner.py` — support DeepSeek gears
- The runner captures `model.forward(ctx)` generically — it already works for any backend.
  Make it construct the model's GRAPH backend via a `model.make_graph_backend(...)` hook
  (Qwen2 → GraphGqaBackend, DeepSeek → GraphMlaBackend), instead of hardcoding GraphGqaBackend.
- Padding: the gear's static buffers already pad to gear size (scratch rows). MoE operates on
  the padded token count (fixed) → static shapes; padding rows route to some expert and are
  discarded at output (as decode-only padding already is). Confirm the MoE path tolerates
  padding rows (they're real tokens computed then ignored — fine).
- `hout`/logits/sampling unchanged (DeepSeek `logits` + batched sample).

### models/deepseek_v2.py + qwen2.py
- Add `make_graph_backend(num_blocks, block_size)` returning (GraphMlaBackend/GraphGqaBackend,
  kv_caches) — parallels the existing `make_attention_backend`.

## Verification (npu2, DeepSeek-V2-Lite) — NPU-only
- **Graph == eager parity**: a `verify_deepseek_graphdecode` (mirror
  verify_qwen2_graphdecode_batched): graph-decode output argmax == eager reference. MUST match.
- **Coherent**: `run_deepseek_chat.py` via the graph executor → coherent (not garbage).
- **Speedup**: `bench_deepseek_decode.py` graph vs eager — expect a large decode tok/s gain.
- **No regression**: Qwen2 graph decode still works (shared runner), host suite, parity_deepseek.
- Host-testable: gear/backend selection logic (pure Python).

## Risks
- MoE ACL-graph capture is NEW here — `npu_moe_*` ops may have graph-capture constraints
  (workspace, dynamic internals). If an MoE op refuses capture, that's the blocker → report
  with the op + error (like the opt1/absorbed findings); consider capturing only attention and
  leaving MoE eager (partial win) or padding differently.
- Keep the SP DeepSeek rope fix (`_ds_rope_interleave`) in the graph MLA path — regressing it
  reintroduces the garbage-output bug.
- Graph capture is finicky (per SP3): expect NPU debug iterations.
