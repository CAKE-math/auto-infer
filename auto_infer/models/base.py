"""Shared decoder forward skeleton for causal LMs (Qwen2/Qwen3 GQA, DeepSeek
MLA+MoE).

`BaseCausalLM` holds the ONE-pass `forward(ctx)` shell + `logits` /
`forward_dense` / `hidden_dense` / `run_layer_dense` / `layer_prefix` that every
model shares; attention is the injected `ctx.attn_backend` so this shell serves
eager-paged / ACL-graph / dense without being rewritten per model.

MODEL CONTRACT a subclass must implement:

  # per-layer hooks used inside forward(ctx)
  * `_compute_cos_sin(positions) -> (cos, sin)`   — RoPE tables (NeoX vs YaRN)
  * `_ffn(i, x, prefix, ctx) -> (T, hidden)`      — per-layer FFN (swiglu vs MoE)

  * `ATTENTION_FAMILY` — key resolved by `layers.attention.registry`

  # construction
  * `from_pretrained(path, device, dtype, ...) -> model`  — stream weights into `self.w`

`__init__` owns `cfg`, `w` (flat HF-name->tensor dict), `device`, `dtype`,
`n_q_local`/`n_kv_local` (TP-local head counts).

Adding a model = (1) a `<Config>` + `<Model>(BaseCausalLM)` in models/<name>.py
implementing the contract above, (2) register in models/registry.py, and — only
if the attention shape is new (not GQA/MLA) — (3) register a backend family in
`layers/attention/registry.py`. Runners never call model-owned backend factories.
"""
import torch

from auto_infer.layers.norm import add_rms_norm as _add_rms_norm
from auto_infer.layers.norm import rms_norm as _rms_norm


class BaseCausalLM:
    USES_FORWARD_CONTEXT = True
    ATTENTION_FAMILY = None
    #: per-layer weight-name prefix (model/checkpoint convention). Fed to the
    #: attention backends + FFN so they index `self.w` by name.
    LAYER_PREFIX = "model.layers.{}."

    def layer_prefix(self, i: int) -> str:
        return self.LAYER_PREFIX.format(i)

    # --- per-layer hooks (used inside forward(ctx)) -------------------------
    def _compute_cos_sin(self, positions):
        """(T,) positions -> (cos, sin), each (T, rope_dim)."""
        raise NotImplementedError

    def _ffn(self, i, x, prefix, ctx):
        """This layer's FFN sub-block: (T, hidden) -> (T, hidden)."""
        raise NotImplementedError

    def prepare_packed_projections(self) -> None:
        """Optionally replace checkpoint projections with execution-ready forms."""
        return None

    @classmethod
    def from_pretrained(cls, path, device, dtype=torch.bfloat16, **_):
        """Stream checkpoint weights into a new model's `self.w` (sharded loader)."""
        raise NotImplementedError

    # --- shared forward skeleton -------------------------------------------
    @torch.no_grad()
    def forward_with_prenorm(self, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """embed -> per-layer [input-norm -> `ctx.attn_backend.attention` ->
        fused residual -> post-norm -> `_ffn`] -> final norm. Returns hidden
        (T, hidden); call :meth:`logits` for (T, vocab). `prenorm=True` returns
        the residual stream BEFORE the final RMSNorm — the raw last-layer hidden
        a DeepSeek/MiMo MTP head consumes (it applies its own hidden_layernorm)."""
        cfg = self.cfg
        be = ctx.attn_backend
        h = self.w["model.embed_tokens.weight"][ctx.token_ids]
        cos, sin = self._compute_cos_sin(ctx.positions)
        ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)   # backends read ctx.cos/sin

        residual = None
        for i in range(cfg.num_layers):
            prefix = self.layer_prefix(i)
            if residual is None:                          # first layer: no incoming residual
                residual = h
                x = _rms_norm(h, self.w[prefix + "input_layernorm.weight"], cfg.rms_eps)
            else:                                         # fused residual-add + input norm
                x, residual = _add_rms_norm(
                    h, residual, self.w[prefix + "input_layernorm.weight"], cfg.rms_eps)

            o = be.attention(i, x, ctx)                    # whole attention sub-block

            x, residual = _add_rms_norm(                    # fused residual-add + post-attn norm
                o, residual, self.w[prefix + "post_attention_layernorm.weight"], cfg.rms_eps)
            h = self._ffn(i, x, prefix, ctx)

        h_norm, h_raw = _add_rms_norm(h, residual, self.w["model.norm.weight"], cfg.rms_eps)
        return h_norm, h_raw

    @torch.no_grad()
    def forward(self, ctx, prenorm=False) -> torch.Tensor:
        h_norm, h_raw = self.forward_with_prenorm(ctx)
        return h_raw if prenorm else h_norm

    def logits(self, hidden: torch.Tensor, out: torch.Tensor | None = None,
               precision: str | None = None) -> torch.Tensor:
        """Project hidden states without retaining a full precision-expanded head.

        The serving path uses the resident model dtype. ``precision="float32"``
        is an explicit, transient reference path for parity diagnostics; unlike
        the old implementation it never stores a vocabulary-sized FP32 copy.
        """
        weight = self.w["lm_head.weight"]
        if precision not in (None, "model", "float32"):
            raise ValueError(f"unsupported logits precision: {precision}")
        if precision == "float32":
            hidden = hidden.float()
            weight = weight.float()
        if out is not None:
            return torch.mm(hidden, weight.t(), out=out)
        return hidden @ weight.t()

    def _dense_ctx(self, token_ids, positions):
        """Single-sequence full-softmax (no-paging) ForwardContext for bring-up /
        parity / MTP drafting."""
        from auto_infer.forward_context import ForwardContext
        from auto_infer.layers.attention.registry import build_attention_backend
        be, caches = build_attention_backend(self, "dense")
        T = token_ids.shape[0]
        return ForwardContext(
            token_ids=token_ids, positions=positions,
            slot_mapping=torch.zeros(T, dtype=torch.int32, device=self.device),
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=self.device),
            cu_seqlens_q=[T], seqlens_kv=[T], attn_mask=None,
            attn_backend=be, kv_caches=caches, is_decode=False)

    @torch.no_grad()
    def forward_dense(self, token_ids, positions):
        """Bring-up / HF-parity: run one sequence through the unified
        forward(ctx) with a full-softmax dense backend (no paging) — single
        source of truth, same layer stack as forward(ctx). Returns (T, vocab)."""
        return self.logits(self.forward(self._dense_ctx(token_ids, positions)))

    @torch.no_grad()
    def hidden_dense(self, token_ids, positions, prenorm=False):
        """Like forward_dense but returns the hidden state (T, hidden) instead of
        logits — the running state a spec-decode MTP head drafts from. Pass
        prenorm=True for the pre-final-norm residual stream (what MTP consumes)."""
        return self.forward(self._dense_ctx(token_ids, positions), prenorm=prenorm)

    @torch.no_grad()
    def run_layer_dense(self, x, positions, layer_idx):
        """One decoder layer's body (input-norm -> dense attention -> fused
        residual -> post-norm -> FFN) over a packed (T, hidden) input, full-
        causal dense (no paging). Reused as the DeepSeek-MTP `decoder_block`
        (spec-decode MTP proposer) — the MTP head is architecturally one shared
        decoder layer. Returns (T, hidden)."""
        from auto_infer.forward_context import ForwardContext
        cfg = self.cfg
        T = x.shape[0]
        from auto_infer.layers.attention.registry import build_attention_backend
        be, caches = build_attention_backend(self, "dense")
        cos, sin = self._compute_cos_sin(positions)
        ctx = ForwardContext(
            token_ids=positions, positions=positions,
            slot_mapping=torch.zeros(T, dtype=torch.int32, device=self.device),
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=self.device),
            cu_seqlens_q=[T], seqlens_kv=[T], attn_mask=None,
            attn_backend=be, kv_caches=caches, is_decode=False)
        ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        prefix = self.layer_prefix(layer_idx)
        residual = x
        xn = _rms_norm(x, self.w[prefix + "input_layernorm.weight"], cfg.rms_eps)
        o = be.attention(layer_idx, xn, ctx)
        xa, residual = _add_rms_norm(
            o, residual, self.w[prefix + "post_attention_layernorm.weight"], cfg.rms_eps)
        # complete residual stream (main loop defers the FFN add to the next layer's
        # fused add_rms_norm; a STANDALONE layer must add it here)
        return residual + self._ffn(layer_idx, xa, prefix, ctx)

    @torch.no_grad()
    def forward_dense_batch(self, seqs):
        """Batched bring-up/eval forward: pack many token-id sequences into ONE
        variable-length forward. ``cu_seqlens_q`` splits them into independent
        causal segments (the dense backend honors it — no cross-sequence
        attention), and RoPE restarts per segment (positions are per-seq
        aranges). Returns ``(hidden (sum_len, hidden_dim), bounds)`` where
        ``bounds[i] = (start, end)`` is sequence i's span on the packed axis.
        Returns HIDDEN, not logits — mc scoring needs only a few positions'
        logits, so the caller runs :meth:`logits` on just those rows instead of
        paying the full (sum_len × vocab) lm_head matmul for tokens it discards."""
        from auto_infer.forward_context import ForwardContext
        dev = self.device
        lens = [len(s) for s in seqs]
        token_ids = torch.tensor([t for s in seqs for t in s], dtype=torch.long, device=dev)
        positions = torch.cat([torch.arange(L, dtype=torch.long, device=dev) for L in lens])
        cu, acc = [], 0
        for L in lens:
            acc += L
            cu.append(acc)
        T = token_ids.shape[0]
        from auto_infer.layers.attention.registry import build_attention_backend
        be, caches = build_attention_backend(self, "dense")
        ctx = ForwardContext(
            token_ids=token_ids, positions=positions,
            slot_mapping=torch.zeros(T, dtype=torch.int32, device=dev),
            block_table=torch.zeros(1, 1, dtype=torch.int32, device=dev),
            cu_seqlens_q=cu, seqlens_kv=list(lens), attn_mask=None,
            attn_backend=be, kv_caches=caches, is_decode=False)
        hidden = self.forward(ctx)
        bounds, s = [], 0
        for L in lens:
            bounds.append((s, s + L))
            s += L
        return hidden, bounds
