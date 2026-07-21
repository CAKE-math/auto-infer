# SP5 — DeepSeek migration + AttentionBackend generalization (skeleton refactor 5/5)

Status: Draft · Date: 2026-07-17 · Branch: feat/skeleton-sp5 · Target: npu2
Verify model: `/data1/models/DeepSeek-V2-Lite-Chat` (DeepseekV2ForCausalLM, MLA + MoE)

## Context & the finding this SP forces
SP1's `AttentionBackend` is `write_kv(k,v)` + `attn(q,k,v)->out` — a **GQA-shaped** interface
(model projects q/k/v, backend does softmax). **MLA does not fit it**: DeepSeek's attention
(`layers/attention/mla.py:mla_paged`) does q_a/q_b + kv_a/kv_b LoRA projections, nope/rope
split, YaRN rope, latent KV, AND o_proj — all INSIDE the block, taking normed `x` + weights,
returning the o_proj'd output. So SP5 can't just "add an MLA backend" to the current
interface — it must **generalize the backend boundary to the whole attention sub-block**.

This is the culminating skeleton step: proving a second, structurally different model (MLA +
MoE) runs on the SAME `forward(ctx)` shell via an injected backend. If MLA fits cleanly, the
skeleton generalizes; the generalization also makes future models (Qwen3 qk-norm, GLM, etc.)
a matter of writing a backend + layer-graph, not a new forward.

## Design — generalize `AttentionBackend` to an attention-block boundary
Change the backend interface from softmax-only to the whole attention sublayer:

```python
class AttentionBackend(ABC):
    def alloc_kv_caches(self, num_blocks, block_size) -> list: ...
    # x: normed hidden (T, hidden); lp: this layer's params; returns the attention
    # sublayer output READY to add to the residual (projected + o_proj'd + tp-reduced).
    # The backend owns: q/k/v (or MLA latent) projection, RoPE, KV write, paged attn, o_proj.
    def attention(self, layer_idx, x, lp, ctx) -> Tensor: ...
```
- `GqaFIABackend` (rename/replace PagedFIABackend): does qkv proj (+bias) from `lp`, RoPE
  (cos/sin from `ctx`), `write_kv`, `paged_fia`, reshape, o_proj, `tp_all_reduce`. This is
  exactly today's Qwen2 forward attention body — MOVED into the backend. Qwen2's `forward(ctx)`
  shrinks to: norm → `be.attention(i, x, lp, ctx)` → residual → norm → mlp.
- `GraphGqaBackend`: the SP3 NZ/graph-capturable variant of the same block.
- `MlaFIABackend` (new): wraps `mla_paged` / `mla_paged_absorbed` (`layers/attention/mla.py`),
  reading DeepSeek MLA params from `lp`; YaRN rope from `ctx`/config. Returns o_proj'd output.
- `DenseBackend`: adapt to the new signature (does the GQA proj too, full-softmax).
- `ForwardContext` gains `cos`/`sin` (or a `rope` handle) so backends share rope tables the
  model computes once. MLA's YaRN tables vs GQA's rope differ — the backend computes/uses the
  right one from config; keep the per-model rope helper.

## Migrate models onto the shared `forward(ctx)` shell
- `models/qwen2.py`: `forward(ctx)` becomes the generic shell (embed → per-layer [norm →
  `be.attention` → residual → norm → mlp] → final norm). Behavior MUST stay bitwise-identical
  (Qwen2 parity gate). The GQA attention body moves verbatim into `GqaFIABackend`.
- `models/deepseek_v2.py`: add `USES_FORWARD_CONTEXT=True`; structured DeepSeek layer params
  (MLA proj weights + per-layer MoE/dense selection); `forward(ctx)` = the same shell but the
  MLP step dispatches dense-`_mlp` vs `_moe` per layer (DeepSeek's existing MoE paths reused).
  Keep `_forward_paged_legacy` (rename current `forward_paged`) as the SP5 parity reference,
  delete after NPU parity. Delete DeepSeek's naive `forward` (→ DenseBackend) or keep a
  `forward_dense` shim like Qwen2.
- Ideally the `forward(ctx)` shell is SHARED (a small base/mixin) so Qwen2 and DeepSeek don't
  each copy it — but keep per-model where the MLP/MoE branch differs. Minimize duplication.

## Verification (npu2)
- **Qwen2 unchanged**: parity (unified vs still-bitwise), smoke_qwen2 HF MATCH, prefix/preempt/
  graphdecode — ALL still green (the GQA-body move into the backend must be behavior-preserving).
- **DeepSeek migrated**: new `tools/parity_deepseek.py` (or extend parity) — unified
  DeepSeek `forward(ctx)`+MlaFIABackend vs `_forward_paged_legacy`, argmax match + max|Δ|.
  Run `smoke_engine_deepseek_paged.py` / `smoke_deepseek_v2.py` on DeepSeek-V2-Lite: coherent
  output. Both absorbed and non-absorbed MLA paths if wired.
- Host suite green (DeepSeek forward is NPU-only for the MLA ops, but structured-params +
  layer-graph + shell logic host-testable).

## Files
- `layers/attention/backend.py` (generalize interface + GqaFIABackend/GraphGqaBackend/
  MlaFIABackend/DenseBackend), `worker/forward_context.py` (+rope), `models/qwen2.py` (shell +
  move GQA body out), `models/deepseek_v2.py` (migrate), `worker/model_runner.py` +
  `graph_decode_runner.py` (construct the right backend per model), tools/parity_deepseek.py.

## Risk / honesty
- This reworks SP1's core interface — the Qwen2 bitwise parity gate is the guardrail (if the
  GQA-body move changes numerics, it's caught).
- MLA on NPU (FIA asymmetric head dims, latent layout) is finicky — the DeepSeek legacy
  forward_paged WORKS (`smoke_engine_deepseek_paged` passes), so migrate by moving its body
  into MlaFIABackend, not rewriting.
- If the generalization proves too invasive to land cleanly in one pass, split: (5a) generalize
  interface + migrate Qwen2 (bitwise gate); (5b) add MlaFIABackend + migrate DeepSeek.
