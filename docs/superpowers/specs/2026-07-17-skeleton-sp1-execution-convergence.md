# SP1 — Execution-layer convergence (skeleton refactor, part 1/5)

Status: Draft · Date: 2026-07-17 · Branch: feat/skeleton-sp1 · Target: Ascend NPU (npu2)

## Context & goal

auto-infer's model↔execution boundary grew organically: the SAME Qwen2 layer stack is
hand-written 3–4× (`forward` naive, `forward_paged`, `forward_cp`, and again inside
`graph_decode_runner.Qwen2GraphAdapter`). This is the #1 model-adaptation cost, a
correctness hazard, and it blocks uniform perf work. This is part 1 of a 5-part skeleton
refactor toward "an agent can adapt a new model by writing ONE file, and the runtime
applies paging/graph/quant/async uniformly for extreme single-model NPU perf."

**SP1 collapses the single-card forwards into ONE imperative `forward(ctx)` driven by an
injected `AttentionBackend` + `ForwardContext`, with structured per-layer weights, a
layer-by-layer parity harness, and dead-code removal — behind the existing `Executor`
interface so `engine/`+`scheduler` are untouched, and verified output-identical to the
current `forward_paged` on NPU.**

Design decisions (confirmed with user):
- Model definition = **imperative `forward(ctx)` calling shared primitives**, execution
  mode injected via `ctx.attn_backend` (vLLM-style). NOT a declarative op-list.
- Full vLLM async is the eventual target (SP4); SP1 only lays the backend/context seam.

## Scope (SP1 only)

IN:
- `AttentionBackend` ABC + `PagedFIABackend` (wraps today's paged FIA path) + `DenseBackend`
  (full causal attn, for bring-up/HF parity — replaces the naive `forward`).
- `ForwardContext` dataclass (per-step inputs + injected backend + kv caches).
- Structured per-layer weights (`Qwen2LayerParams`) resolved once at load; keep `model.w`
  dict available too (so `graph_decode_runner` is untouched this SP).
- Single `Qwen2Model.forward(ctx)` using shared primitives; remove naive `forward` and
  `forward_paged` (behavior moves into forward(ctx)+backend).
- Layer-by-layer parity harness `tools/parity.py`.
- Delete dead seams: `worker/streams.py` (StreamManager), `worker/npu_graph.py`
  (DecodeGraph, superseded). Add a README note that `spec_decode/` is unwired.
- `NpuModelRunner` builds `ForwardContext` (still per-step; persistence is SP2) with a
  `PagedFIABackend`; `PagedNpuExecutor` public interface unchanged.

OUT (later SPs, do NOT touch):
- Persistent/incremental input buffers → SP2.
- Graph-capture default decode → SP3 (`graph_decode_runner` left AS-IS this SP).
- Worker-thread futures + output-thread D2H (full vLLM async) → SP4.
- DeepSeek migration to the new skeleton → SP5.
- `forward_cp` (context parallel): **left untouched** this SP; re-expressed as a
  `ContextParallelBackend` in a later SP. (So Qwen2 temporarily still has `forward_cp`
  alongside the unified `forward`; that's accepted — the headline 3-way collapse
  (naive + paged + graph-adapter-need) is what SP1 delivers.)

## Interfaces

### `layers/attention/backend.py`
```python
class AttentionBackend(ABC):
    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list: ...
    def write_kv(self, layer_idx: int, k: Tensor, v: Tensor, ctx: "ForwardContext") -> None: ...
    # q,k,v are (T, n_heads/n_kv, head_dim) post-RoPE; returns attention out (T, n_q, hd)
    def attn(self, layer_idx: int, q: Tensor, k: Tensor, v: Tensor, ctx: "ForwardContext") -> Tensor: ...

class PagedFIABackend(AttentionBackend):
    # canonical KV layout (2, num_blocks, block_size, n_kv, head_dim); write via write_kv
    # (paged.py), attn via paged_fia (paged.py). Exactly today's forward_paged attention.
    def __init__(self, n_q_heads, n_kv_heads, head_dim, scale): ...

class DenseBackend(AttentionBackend):
    # full O(T^2) causal softmax (fp32), no paging. alloc_kv_caches returns []; write_kv
    # is a no-op; attn recomputes over the whole (T) sequence. For bring-up/HF parity only.
```

### `worker/forward_context.py`
```python
@dataclass
class ForwardContext:
    token_ids: Tensor            # (T,) long
    positions: Tensor           # (T,) long
    slot_mapping: Tensor        # (T,) int32  (paged; unused by DenseBackend)
    block_table: Tensor         # (num_reqs, max_blocks) int32
    cu_seqlens_q: list[int]     # cumulative query lengths (TND)
    seqlens_kv: list[int]       # per-request kv length (TND)
    attn_mask: Tensor           # causal template / bool mask
    attn_backend: AttentionBackend
    kv_caches: list             # per-layer, from attn_backend.alloc_kv_caches
    is_decode: bool
```

### `models/qwen2.py` (single forward)
```python
@dataclass
class Qwen2LayerParams:
    input_ln: Tensor; post_ln: Tensor
    q_w; q_b; k_w; k_b; v_w; v_b; o_w
    gate_w; up_w; down_w
    # W8A8: *_w may be a (int8, scale) tuple (existing _lin dispatch preserved)

class Qwen2Model:
    self.layers: list[Qwen2LayerParams]     # built at load from HF names
    self.w: dict                             # KEPT (graph_decode_runner reads it)
    def forward(self, ctx: ForwardContext) -> Tensor:   # ONE forward, returns hidden (T,hidden)
    def logits(self, hidden: Tensor) -> Tensor:         # (T, vocab) fp32; cached fp32 lm_head weight
```
Shared primitives live in `layers/` (rms_norm, add_rms_norm, rope, swiglu already exist;
add `qkv_proj`, `decoder_layer` helpers or inline in forward using LayerParams).

### `worker/model_runner.py`
`NpuModelRunner.__init__` builds `PagedFIABackend` + `kv_caches` via the backend.
`_build(...)` returns a `ForwardContext` (same marshaling as today, now packaged in the
dataclass + backend). `submit`/`sampled_of`/`collect` unchanged (batched sampling stays).

### `tools/parity.py`
```
python tools/parity.py <model_path>
  # loads Qwen2; runs a fixed prompt through the NEW forward(ctx) [PagedFIABackend]
  # and a REFERENCE, layer-by-layer; prints first layer whose max|Δ| > tol, else PASS.
```
Reference for SP1 = the legacy `forward_paged` (kept temporarily as `_forward_paged_legacy`
during the refactor, diffed per-layer, then deleted in the final task). Also keeps the
HF-logits parity check from `smoke_qwen2`.

## Verification (npu2, container `auto-infer-dev-20260624`, Qwen2.5-0.5B)
- **Host suite green** (in-container `pytest -q`).
- **Per-layer parity**: new `forward(ctx)` [PagedFIABackend] vs legacy `forward_paged`,
  max|Δ| within tol at every layer + final logits argmax identical.
- **`smoke_qwen2` HF parity** still MATCH (via DenseBackend path or paged).
- **`verify_prefix_cache` + `verify_preemption`** still PASS (engine path unchanged).
- **`verify_qwen2_graphdecode_batched`** still PASS (graph_decode_runner untouched).

## Files
- Create: `layers/attention/backend.py`, `worker/forward_context.py`, `tools/parity.py`
- Modify: `models/qwen2.py` (structured layers + single forward), `models/loader.py`
  (build LayerParams), `worker/model_runner.py` (ForwardContext + backend)
- Delete: `worker/streams.py`, `worker/npu_graph.py`
- Untouched: `engine/*`, `scheduler`, `graph_decode_runner.py`, `deepseek_v2.py`,
  `qwen2.forward_cp`, `layers/attention/{paged,mla}.py`

## Task order (TDD, subagent-driven)
1. `AttentionBackend` ABC + `PagedFIABackend` + `DenseBackend` (host-unit where possible;
   NPU parity of PagedFIABackend.attn vs `paged_fia` deferred to the integration parity).
2. `ForwardContext` dataclass + `NpuModelRunner._build` returns it (behavior-preserving).
3. Structured `Qwen2LayerParams` + loader builder (keep `model.w`); host test on name mapping.
4. Single `Qwen2Model.forward(ctx)`; keep `_forward_paged_legacy`; host: dense forward vs
   a tiny reference.
5. `tools/parity.py` + NPU per-layer parity new-vs-legacy (npu2).
6. Delete legacy `forward_paged`, `streams.py`, `npu_graph.py`; README note; final NPU
   regression (smoke + prefix + preempt + graphdecode).
