"""Routed + shared-expert MoE sub-block (DeepSeek V2/V3).

Mirrors the `AttentionBackend` seam: the model composes this block and calls
`moe(x, layer_idx)` for its MoE layers, exactly as it calls
`attn_backend.attention(i, x, ctx)` for the attention sub-block. The block
owns everything MoE:
  * gating/routing — softmax|sigmoid scoring + optional group-limited
    (`noaux_tc`, DeepSeek-V3) top-k selection with `e_score_correction_bias`;
  * expert compute — grouped-GEMM fused (ep=1), fused Expert-Parallel, or the
    naive per-expert loop fallback;
  * sequence-parallel (SP) token sharding across the TP group;
  * per-layer stacked-expert-weight caching (built once, originals freed).

`cfg` is duck-typed: any object exposing `n_routed`, `top_k`, `scoring_func`,
`topk_method`, `n_group`, `topk_group`, `norm_topk_prob`, `routed_scale` works.
`w` is held BY REFERENCE (fused compute pops the per-expert originals out of it
to avoid 2x expert memory), same as the attention backends share `model.w`.
"""
import torch

from auto_infer.layers.mlp import swiglu_mlp


class MoE:
    def __init__(self, w, cfg, device, dtype, fused=True, free_originals=True,
                 layer_prefix=None):
        self.w = w
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.fused = fused              # grouped-GEMM path (ep=1/EP) vs naive loop
        self.free_originals = free_originals   # drop per-expert originals after stacking
        # per-layer weight-name prefix is a model/checkpoint convention, owned by
        # the model (passed from from_pretrained) — not hardcoded in the block.
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self._fused_w = {}              # per-layer (w13, w2) stacked-weight cache (ep=1)
        self._fused_w_ep = {}           # per-layer cache for the EP path
        self._ep_dispatch = None

    def __call__(self, x, layer_idx, active_token_mask=None):
        """SP wrapper (used by DeepSeek-V3). When SP is on, shard tokens along the
        sequence dim across the TP group so each rank runs gate+experts on only its
        1/sp tokens, then all-gather back. MoE is per-token independent, so SP is
        numerically identical to non-SP."""
        from auto_infer.distributed.parallel_state import sp_all_gather, sp_chunk, sp_size
        if sp_size() > 1:
            x_local, num_tokens = sp_chunk(x)
            mask_local = None
            if active_token_mask is not None:
                mask_local, mask_tokens = sp_chunk(active_token_mask)
                if mask_tokens != num_tokens:
                    raise ValueError("MoE active-token mask does not match input")
            return sp_all_gather(
                self._compute(x_local, layer_idx, mask_local), num_tokens)
        return self._compute(x, layer_idx, active_token_mask)

    def _gate(self, router, p):
        """Routed gating. V2-Lite: softmax + greedy top-k. DeepSeek-V3: sigmoid
        scoring + group-limited (noaux_tc) selection w/ e_score_correction_bias.
        Config-selected (scoring_func / topk_method)."""
        cfg, w = self.cfg, self.w
        T = router.shape[0]
        scores = router.sigmoid() if cfg.scoring_func == "sigmoid" else router.softmax(dim=-1)
        if cfg.topk_method == "noaux_tc":
            bias = w.get(p + "gate.e_score_correction_bias")
            choice = scores + bias.float() if bias is not None else scores
            ng, tg = cfg.n_group, cfg.topk_group
            grp = choice.view(T, ng, -1)
            gscore = grp.topk(2, dim=-1).values.sum(-1)               # (T, ng): top-2 per group
            gsel = gscore.topk(tg, dim=-1).indices                    # pick topk_group groups
            gmask = torch.zeros(T, ng, device=router.device, dtype=torch.bool)
            gmask.scatter_(1, gsel, True)
            full = gmask.unsqueeze(-1).expand_as(grp).reshape(T, -1)
            choice = choice.masked_fill(~full, float("-inf"))
            topk_i = choice.topk(cfg.top_k, dim=-1).indices
            topk_w = scores.gather(1, topk_i)                         # weights from sigmoid scores
        else:
            topk_w, topk_i = scores.topk(cfg.top_k, dim=-1)
        if cfg.norm_topk_prob:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
        return topk_w, topk_i

    def _compute(self, x, i, active_token_mask=None):
        """Select single-rank grouped GEMM or fused distributed EP."""
        from auto_infer.distributed.parallel_state import ep_size
        if self.fused:
            return (self._fused_compute(x, i) if ep_size() == 1
                    else self._fused_ep(x, i, active_token_mask))
        if ep_size() > 1:
            raise RuntimeError("EP requires fused BF16 dispatch/combine")
        return self._naive(x, i)

    def _fused_compute(self, x, i):
        """Grouped-GEMM fused routed experts — single-rank (ep=1) path: all experts
        local, one permute + grouped GEMM instead of a per-expert loop. Numerically
        equivalent to _naive. Weights stacked once per layer and cached."""
        from auto_infer.layers.moe.fused_moe import build_expert_weights, fused_experts
        cfg, w = self.cfg, self.w
        p = self.layer_prefix(i) + "mlp."
        if i not in self._fused_w:
            self._fused_w[i] = build_expert_weights(w, p, cfg.n_routed)
            if self.free_originals:                      # drop per-expert originals to
                for e in range(cfg.n_routed):            # avoid 2x expert memory (fused-only)
                    for nm in ("gate_proj", "up_proj", "down_proj"):
                        w.pop(f"{p}experts.{e}.{nm}.weight", None)
        w13, w2 = self._fused_w[i]
        router = (x @ w[p + "gate.weight"].t()).float()
        topk_w, topk_i = self._gate(router, p)
        topk_w = (topk_w * cfg.routed_scale).to(self.dtype)
        out = fused_experts(x, topk_i, topk_w, w13, w2, cfg.n_routed)
        return out + swiglu_mlp(x, w, p + "shared_experts.")

    def _fused_ep(self, x, i, active_token_mask=None):
        """True EP: fused token dispatch, local grouped GEMM, fused combine."""
        from auto_infer.distributed.parallel_state import ep_rank, ep_size, ep_topology
        from auto_infer.layers.moe.ep_dispatch import NpuMoeDispatchCombine
        from auto_infer.layers.moe.fused_moe import (
            build_expert_weights, fused_local_experts)
        cfg, w = self.cfg, self.w
        p = self.layer_prefix(i) + "mlp."
        ep, r = ep_size(), ep_rank()
        if cfg.n_routed % ep:
            raise ValueError("global expert count must be divisible by EP size")
        n_local = cfg.n_routed // ep
        lo = r * n_local
        if i not in self._fused_w_ep:
            self._fused_w_ep[i] = build_expert_weights(w, p, cfg.n_routed, lo, lo + n_local)
            if self.free_originals:
                for e in range(lo, lo + n_local):
                    for nm in ("gate_proj", "up_proj", "down_proj"):
                        w.pop(f"{p}experts.{e}.{nm}.weight", None)
        w13, w2 = self._fused_w_ep[i]
        router = (x @ w[p + "gate.weight"].t()).float()
        topk_w, topk_i = self._gate(router, p)
        topk_w = (topk_w * cfg.routed_scale).to(self.dtype)
        topk_i = topk_i.to(torch.int32)
        if self._ep_dispatch is None:
            self._ep_dispatch = NpuMoeDispatchCombine(
                ep_topology(), cfg.n_routed, self.dtype)
        metadata = self._ep_dispatch.dispatch(
            x, topk_i, active_token_mask)
        local = fused_local_experts(
            metadata.hidden_states, metadata.expert_tokens, w13, w2)
        out = self._ep_dispatch.combine(
            local, topk_i, topk_w, metadata, active_token_mask)
        return out + swiglu_mlp(x, w, p + "shared_experts.")

    def _naive(self, x, i):
        cfg, w = self.cfg, self.w
        p = self.layer_prefix(i) + "mlp."
        router = (x @ w[p + "gate.weight"].t()).float()           # (T, n_routed)
        topk_w, topk_i = self._gate(router, p)
        topk_w = (topk_w * cfg.routed_scale).to(self.dtype)
        out = torch.zeros_like(x)
        for slot in range(cfg.top_k):
            eidx = topk_i[:, slot]
            wgt = topk_w[:, slot].unsqueeze(-1)
            for e in eidx.unique().tolist():
                mask = eidx == e
                out[mask] += wgt[mask] * swiglu_mlp(x[mask], w, f"{p}experts.{e}.")
        out = out + swiglu_mlp(x, w, p + "shared_experts.")
        return out
