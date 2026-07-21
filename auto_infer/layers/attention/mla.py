"""MLA (Multi-head Latent Attention) building block for DeepSeek V2/V3.

The reusable MLA paged-attention block that models compose. DeepSeek's Q/KV go
through low-rank latent projections (q_a/q_b when q_lora_rank set, else q_proj;
kv_a_proj_with_mqa + kv_a_layernorm + kv_b_proj), split into nope/rope parts, YaRN
RoPE applied to the pe parts; K/V are written to the paged cache and attended via the
shared FIA op. Uses the shared rope/norm primitives so it does not depend on any
model module.

This file has two forms: non-absorbed (K=nope+rope, V=vd materialized per head) and
absorbed-latent (a single MQA-style latent head, ~9x smaller cache).
"""
import torch

from auto_infer.layers.attention.base import (
    AttentionBackend, dense_causal_attention, paged_fia, write_kv)
from auto_infer.layers.attention.graph_fia import GraphFiaLifecycle
from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.layers.rotary_embedding import ds_rope_interleave as _ds_rope_interleave
from auto_infer.layers.rotary_embedding import rotate_half as _rotate_half


def mla_paged(x, w, prefix, *, num_heads, qk_nope, qk_rope, v_head_dim, kv_lora_rank,
              q_lora_rank, rms_eps, cos1, sin1, k_cache, v_cache, slot_mapping,
              block_table, block_size, actual_seq_q, actual_seq_kv, attn_mask,
              softmax_scale):
    """One MLA layer's paged attention. x: (T, hidden); w: weight dict; prefix: layer
    prefix 'model.layers.{i}.'. cos1/sin1: (T,1,rope) YaRN rope tables. Returns the
    attention output projected by o_proj: (T, hidden)."""
    T = x.shape[0]
    nh, nope, rope, vd, qk = num_heads, qk_nope, qk_rope, v_head_dim, qk_nope + qk_rope

    if q_lora_rank is None:
        q = x @ w[prefix + "self_attn.q_proj.weight"].t()
    else:
        q = _rms_norm(x @ w[prefix + "self_attn.q_a_proj.weight"].t(),
                      w[prefix + "self_attn.q_a_layernorm.weight"], rms_eps)
        q = q @ w[prefix + "self_attn.q_b_proj.weight"].t()
    q = q.view(T, nh, qk)
    q_nope, q_pe = q.split([nope, rope], dim=-1)

    ckv = x @ w[prefix + "self_attn.kv_a_proj_with_mqa.weight"].t()
    ckv, k_pe = ckv.split([kv_lora_rank, rope], dim=-1)
    k_pe = k_pe.view(T, 1, rope)
    kv = _rms_norm(ckv, w[prefix + "self_attn.kv_a_layernorm.weight"], rms_eps)
    kv = (kv @ w[prefix + "self_attn.kv_b_proj.weight"].t()).view(T, nh, nope + vd)
    k_nope, v = kv.split([nope, vd], dim=-1)

    q_pe = _ds_rope_interleave(q_pe)                # DeepSeek interleaved-rope layout
    k_pe = _ds_rope_interleave(k_pe)
    q_pe = q_pe * cos1 + _rotate_half(q_pe) * sin1
    k_pe = k_pe * cos1 + _rotate_half(k_pe) * sin1
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(T, nh, rope)], dim=-1).contiguous()
    v = v.contiguous()

    write_kv(k, v, k_cache, v_cache, slot_mapping)
    nb = k_cache.shape[0]
    attn = paged_fia(q, k_cache.view(nb, block_size, nh * qk),
                     v_cache.view(nb, block_size, nh * vd), block_table,
                     block_size=block_size, actual_seq_q=actual_seq_q,
                     actual_seq_kv=actual_seq_kv, num_kv_heads=nh, num_heads=nh,
                     scale=softmax_scale, atten_mask=attn_mask)
    return attn.reshape(T, nh * vd) @ w[prefix + "self_attn.o_proj.weight"].t()


def build_absorb_weights(kv_b_weight, num_heads, qk_nope, v_head_dim, kv_lora_rank):
    """Split kv_b_proj (nh*(nope+vd), kvl) into W_UK (nh, nope, kvl) and
    W_UV (nh, vd, kvl) for the absorbed MLA. W_UK projects the query into latent
    space; W_UV absorbs the value up-projection into the output."""
    W = kv_b_weight.view(num_heads, qk_nope + v_head_dim, kv_lora_rank)
    w_uk, w_uv = W.split([qk_nope, v_head_dim], dim=1)
    return w_uk.contiguous(), w_uv.contiguous()


def mla_paged_absorbed(x, w, prefix, *, num_heads, qk_nope, qk_rope, v_head_dim,
                       kv_lora_rank, q_lora_rank, rms_eps, cos1, sin1, k_cache,
                       v_cache, slot_mapping, block_table, block_size, actual_seq_q,
                       actual_seq_kv, attn_mask, softmax_scale, w_uk, w_uv):
    """Absorbed MLA: caches only the latent ckv[kv_lora]+k_pe[rope] (a single
    MQA-style kv head) instead of full per-head K/V — ~9x smaller KV. Query is
    projected into latent space via W_UK, attention runs over the latent, and W_UV
    is absorbed into the output. Numerically == non-absorbed. k_cache holds
    (kv_lora+rope) per token, v_cache holds kv_lora."""
    import torch
    T = x.shape[0]
    nh, nope, rope, vd, kvl = num_heads, qk_nope, qk_rope, v_head_dim, kv_lora_rank
    if q_lora_rank is None:
        q = x @ w[prefix + "self_attn.q_proj.weight"].t()
    else:
        q = _rms_norm(x @ w[prefix + "self_attn.q_a_proj.weight"].t(),
                      w[prefix + "self_attn.q_a_layernorm.weight"], rms_eps)
        q = q @ w[prefix + "self_attn.q_b_proj.weight"].t()
    q = q.view(T, nh, nope + rope)
    q_nope, q_pe = q.split([nope, rope], dim=-1)
    ckv_full = x @ w[prefix + "self_attn.kv_a_proj_with_mqa.weight"].t()
    ckv, k_pe = ckv_full.split([kvl, rope], dim=-1)
    ckv = _rms_norm(ckv, w[prefix + "self_attn.kv_a_layernorm.weight"], rms_eps)  # (T, kvl)
    k_pe = k_pe.view(T, 1, rope)
    q_pe = _ds_rope_interleave(q_pe)                # DeepSeek interleaved-rope layout
    k_pe = _ds_rope_interleave(k_pe)
    q_pe = q_pe * cos1 + _rotate_half(q_pe) * sin1
    k_pe = k_pe * cos1 + _rotate_half(k_pe) * sin1                            # (T,1,rope)

    q_absorbed = torch.einsum("tnp,npl->tnl", q_nope, w_uk)                   # (T,nh,kvl)
    q_latent = torch.cat([q_absorbed, q_pe], dim=-1).contiguous()            # (T,nh,kvl+rope)
    kd = kvl + rope
    # value = full latent (K); FIA needs symmetric K/V head dims (asymmetric only
    # supports 192/128). attn @ latent restricted to the first kvl dims == attn @ ckv,
    # so we attend with V=K then slice [:kvl] — numerically the absorbed value output.
    k_latent = torch.cat([ckv.unsqueeze(1), k_pe], dim=-1).contiguous()      # (T,1,kd)

    write_kv(k_latent, k_latent, k_cache, v_cache, slot_mapping)
    nb = k_cache.shape[0]
    attn = paged_fia(q_latent, k_cache.view(nb, block_size, kd),
                     v_cache.view(nb, block_size, kd), block_table,
                     block_size=block_size, actual_seq_q=actual_seq_q,
                     actual_seq_kv=actual_seq_kv, num_kv_heads=1, num_heads=nh,
                     scale=softmax_scale, atten_mask=attn_mask)              # (T,nh,kd)
    o_latent = attn.view(T, nh, kd)[..., :kvl]                                # ckv-value part
    o = torch.einsum("tnl,nvl->tnv", o_latent, w_uv)                          # (T,nh,vd)
    return o.reshape(T, nh * vd) @ w[prefix + "self_attn.o_proj.weight"].t()


class _MlaProjMixin:
    """Shared DeepSeek MLA (non-absorbed) sub-block: q/kv low-rank projection +
    nope/rope split + DeepSeek interleaved YaRN RoPE + per-head decompressed K/V
    + o_proj — the part IDENTICAL across eager (plain paged-FIA) and ACL-graph
    (FIA-v2 `.out`) execution. Subclasses supply `_write_kv`/`_attn` (the only op
    pair that differs), mirroring `_GqaProjRopeMixin` for GQA. Needs self.{w,
    layer_prefix, num_heads, qk_nope, qk_rope, v_head_dim, kv_lora_rank,
    q_lora_rank, rms_eps, softmax_scale}."""

    def attention(self, layer_idx: int, x: torch.Tensor, ctx) -> torch.Tensor:
        w = self.w
        T = x.shape[0]
        nh, nope, rope, vd = self.num_heads, self.qk_nope, self.qk_rope, self.v_head_dim
        qk = nope + rope
        cos1, sin1 = ctx.cos, ctx.sin
        p = self.layer_prefix(layer_idx)
        if self.q_lora_rank is None:
            q = x @ w[p + "self_attn.q_proj.weight"].t()
        else:
            q = _rms_norm(x @ w[p + "self_attn.q_a_proj.weight"].t(),
                          w[p + "self_attn.q_a_layernorm.weight"], self.rms_eps)
            q = q @ w[p + "self_attn.q_b_proj.weight"].t()
        q = q.view(T, nh, qk)
        q_nope, q_pe = q.split([nope, rope], dim=-1)
        ckv = x @ w[p + "self_attn.kv_a_proj_with_mqa.weight"].t()
        ckv, k_pe = ckv.split([self.kv_lora_rank, rope], dim=-1)
        k_pe = k_pe.view(T, 1, rope)
        kv = _rms_norm(ckv, w[p + "self_attn.kv_a_layernorm.weight"], self.rms_eps)
        kv = (kv @ w[p + "self_attn.kv_b_proj.weight"].t()).view(T, nh, nope + vd)
        k_nope, v = kv.split([nope, vd], dim=-1)
        q_pe = _ds_rope_interleave(q_pe)                # DeepSeek interleaved-rope layout
        k_pe = _ds_rope_interleave(k_pe)
        q_pe = q_pe * cos1 + _rotate_half(q_pe) * sin1
        k_pe = k_pe * cos1 + _rotate_half(k_pe) * sin1
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(T, nh, rope)], dim=-1).contiguous()
        v = v.contiguous()
        self._write_kv(layer_idx, k, v, ctx)
        o = self._attn(layer_idx, q, k, v, ctx).reshape(T, nh * vd)
        return o @ w[p + "self_attn.o_proj.weight"].t()


class MlaFIABackend(_MlaProjMixin, AttentionBackend):
    """DeepSeek MLA attention sub-block: wraps `mla_paged`/`mla_paged_absorbed`.
    `w` (the model's raw HF-named weight dict) is held by reference at
    construction, and the per-layer weight prefix comes from the model-owned
    `layer_prefix(layer_idx)`; `mla_paged` indexes `w` by name.

    Config (num_heads/qk_nope/qk_rope/v_head_dim/kv_lora_rank/q_lora_rank/
    rms_eps/softmax_scale) is fixed at construction; `ctx.cos`/`ctx.sin` carry the
    YaRN rope tables the model computes once per forward.

    Two attention variants, chosen at construction (`absorb`):
      * absorb=False (default): `mla_paged` — per-head decompressed K/V.
      * absorb=True: `mla_paged_absorbed` — latent-space attention with the
        query absorbed via `W_UK`/`W_UV` (~9x smaller KV cache); the per-layer
        (W_UK, W_UV) split is built lazily and cached (`self._absorb_w`).
    """

    def __init__(self, w: dict, *, num_heads: int, qk_nope: int, qk_rope: int,
                 v_head_dim: int, kv_lora_rank: int, q_lora_rank, rms_eps: float,
                 softmax_scale: float, num_layers: int, device, dtype, absorb: bool = False,
                 layer_prefix=None):
        self.w = w
        self.num_heads = num_heads
        self.qk_nope = qk_nope
        self.qk_rope = qk_rope
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rms_eps = rms_eps
        self.softmax_scale = softmax_scale
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.absorb = absorb
        # per-layer weight-name prefix is a MODEL/checkpoint convention, owned by
        # the model (passed by the attention registry) — not hardcoded here.
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self._absorb_w: dict = {}     # per-layer (W_UK, W_UV) cache for absorbed MLA

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        """Per-layer (k_cache, v_cache): absorbed MLA caches only the latent
        ckv[kv_lora]+k_pe[rope] as one MQA head (~9x smaller); non-absorbed caches
        per-head decompressed K (qk_nope+qk_rope) and V (v_head_dim)."""
        if self.absorb:
            kd = self.kv_lora_rank + self.qk_rope          # symmetric latent K=V head dim
            kshape = vshape = (num_blocks, block_size, 1, kd)
        else:
            qk = self.qk_nope + self.qk_rope
            kshape = (num_blocks, block_size, self.num_heads, qk)
            vshape = (num_blocks, block_size, self.num_heads, self.v_head_dim)
        return [(torch.zeros(kshape, device=self.device, dtype=self.dtype),
                 torch.zeros(vshape, device=self.device, dtype=self.dtype))
                for _ in range(self.num_layers)]

    def attention(self, layer_idx: int, x: torch.Tensor, ctx) -> torch.Tensor:
        if not self.absorb:
            return super().attention(layer_idx, x, ctx)     # shared _MlaProjMixin body
        # absorbed variant (latent-space attention, ~9x smaller KV) — structurally
        # different from non-absorbed, so it stays in layers/attention/mla.py.
        k_cache, v_cache = ctx.kv_caches[layer_idx]
        block_size = k_cache.shape[1]
        prefix = self.layer_prefix(layer_idx)
        nh, nope, rope, vd = self.num_heads, self.qk_nope, self.qk_rope, self.v_head_dim
        if layer_idx not in self._absorb_w:
            self._absorb_w[layer_idx] = build_absorb_weights(
                self.w[prefix + "self_attn.kv_b_proj.weight"], nh, nope, vd, self.kv_lora_rank)
        w_uk, w_uv = self._absorb_w[layer_idx]
        return mla_paged_absorbed(
            x, self.w, prefix, num_heads=nh, qk_nope=nope, qk_rope=rope,
            v_head_dim=vd, kv_lora_rank=self.kv_lora_rank, q_lora_rank=self.q_lora_rank,
            rms_eps=self.rms_eps, cos1=ctx.cos, sin1=ctx.sin, k_cache=k_cache, v_cache=v_cache,
            slot_mapping=ctx.slot_mapping, block_table=ctx.block_table,
            block_size=block_size, actual_seq_q=ctx.cu_seqlens_q,
            actual_seq_kv=ctx.seqlens_kv, attn_mask=ctx.attn_mask,
            softmax_scale=self.softmax_scale, w_uk=w_uk, w_uv=w_uv)

    def _write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, ctx) -> None:
        kc, vc = ctx.kv_caches[layer_idx]
        write_kv(k, v, kc, vc, ctx.slot_mapping)

    def _attn(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ctx) -> torch.Tensor:
        kc, vc = ctx.kv_caches[layer_idx]
        nb, block_size = kc.shape[0], kc.shape[1]
        nh, vd, qk = self.num_heads, self.v_head_dim, self.qk_nope + self.qk_rope
        return paged_fia(q, kc.view(nb, block_size, nh * qk), vc.view(nb, block_size, nh * vd),
                         ctx.block_table, block_size=block_size, actual_seq_q=ctx.cu_seqlens_q,
                         actual_seq_kv=ctx.seqlens_kv, num_kv_heads=nh, num_heads=nh,
                         scale=self.softmax_scale, atten_mask=ctx.attn_mask)


class GraphMlaBackend(_MlaProjMixin, GraphFiaLifecycle, AttentionBackend):
    """ACL-graph-capturable MLA attention: mirrors `GraphGqaBackend`'s capture
    machinery (NZ-layout paged KV via `npu_scatter_pa_kv_cache`, attention via
    FIA-v2 `.out`, `graph_task_group_begin/end` -> handle appended to `self.reg`,
    `update(ctx)` re-issuing with the step's fresh `ctx.seqlens_kv` via
    `graph_task_update_begin/end`, `begin_capture`/`end_capture` toggling
    `capturing`), but the per-layer body is MLA's non-absorbed sub-block (via
    `_MlaProjMixin`). The DeepSeek interleaved-rope reshape (`_ds_rope_interleave`)
    is required for correctness — regressing it produces garbage output. Only the
    KV-write + core-attention op pair is swapped for `GraphGqaBackend`'s
    NZ/FIA-v2 `.out` pair.

    MLA's K and V head dims differ (qk = qk_nope + qk_rope vs v_head_dim), so
    unlike `GraphGqaBackend` (symmetric K/V head_dim) the NZ views for K and V
    are computed independently (each head_dim must be divisible by `NZ`).

    K/V attention here is MHA, not GQA: `mla_paged` calls `paged_fia` with
    `num_kv_heads=num_heads` (one decompressed KV head per query head), so
    FIA-v2 is called the same way (no query/kv head-count mismatch to repeat).

    Absorbed MLA (smaller latent KV cache) is OUT of this graph path: the
    plain-FIA absorbed variant already hits the block_size-128 / head-dim-576 op
    constraint — non-absorbed only.

    No `tp_all_reduce` here, matching `MlaFIABackend.attention` (DeepSeek's MLA
    sub-block does not TP-shard attention in this codebase; `_lin`'s W8A8
    dispatch is likewise not used — `mla_paged`'s weight access is plain matmul
    against `self.w`)."""

    supports_prefill_graph = True
    NZ = 16  # FRACTAL_NZ inner-tile width for npu_scatter_pa_kv_cache / FIA-v2

    def __init__(self, w: dict, *, num_heads: int, qk_nope: int, qk_rope: int,
                 v_head_dim: int, kv_lora_rank: int, q_lora_rank, rms_eps: float,
                 softmax_scale: float, num_layers: int, device, dtype, layer_prefix=None):
        self.w = w
        self.num_heads = num_heads
        self.qk_nope = qk_nope
        self.qk_rope = qk_rope
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rms_eps = rms_eps
        self.softmax_scale = softmax_scale
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self._init_graph_fia()

    @property
    def _fia_query_heads(self):
        return self.num_heads

    @property
    def _fia_kv_heads(self):
        return self.num_heads

    @property
    def _fia_scale(self):
        return self.softmax_scale

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        """Per-layer (k_cache, v_cache) — same as `MlaFIABackend`'s non-absorbed
        branch: per-head decompressed K (qk_nope+qk_rope) and V (v_head_dim). The
        NZ reinterpretation happens via `.view()` at write/attn time (contiguous-
        memory bitcast), not in the allocated shape itself."""
        qk = self.qk_nope + self.qk_rope
        kshape = (num_blocks, block_size, self.num_heads, qk)
        vshape = (num_blocks, block_size, self.num_heads, self.v_head_dim)
        return [(torch.zeros(kshape, device=self.device, dtype=self.dtype),
                 torch.zeros(vshape, device=self.device, dtype=self.dtype))
                for _ in range(self.num_layers)]

    # attention() is inherited from _MlaProjMixin (shared with MlaFIABackend);
    # this class only overrides the KV-write + core-attention op pair below.

    def _write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, ctx) -> None:
        """`npu_scatter_pa_kv_cache` with NZ views (mirrors
        `GraphGqaBackend._write_kv`), scatter by `ctx.slot_mapping`. K and V
        NZ-tile counts computed independently since MLA's per-head K/V dims
        differ (qk = qk_nope+qk_rope vs v_head_dim)."""
        import torch_npu
        kc, vc = ctx.kv_caches[layer_idx]
        num_blocks, block_size = kc.shape[0], kc.shape[1]
        qk = self.qk_nope + self.qk_rope
        torch_npu.npu_scatter_pa_kv_cache(
            k.contiguous(), v.contiguous(),
            kc.view(num_blocks, self.num_heads * qk // self.NZ, block_size, self.NZ),
            vc.view(num_blocks, self.num_heads * self.v_head_dim // self.NZ, block_size, self.NZ),
            ctx.slot_mapping)

    def _attn(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ctx) -> torch.Tensor:
        """FIA-v2 `.out`, capture-aware — same shape as
        `GraphGqaBackend._attn`, but `num_key_value_heads=num_heads` (MLA's
        per-head decompressed K/V is one KV head per query head, matching
        `mla_paged`'s `paged_fia(..., num_kv_heads=nh, num_heads=nh)`)."""
        kc, vc = ctx.kv_caches[layer_idx]
        block_size = kc.shape[1]
        nh, vd = self.num_heads, self.v_head_dim
        qk = self.qk_nope + self.qk_rope
        knz = kc.view(-1, nh, qk // self.NZ, block_size, self.NZ)
        vnz = vc.view(-1, nh, vd // self.NZ, block_size, self.NZ)
        T = q.shape[0]
        o = torch.empty(T, nh, vd, dtype=q.dtype, device=q.device)
        lse = torch.empty(T, dtype=torch.float32, device=q.device)
        self._run_graph_fia(q, knz, vnz, o, lse, ctx, block_size)
        return o


class MlaDenseBackend(_MlaProjMixin, AttentionBackend):
    """Full-softmax MLA (no paging) for bring-up / HF-parity via forward_dense.
    Reuses `_MlaProjMixin`'s q/kv-LoRA proj + interleaved YaRN RoPE; `_attn` is a
    full causal softmax (fp32) over the whole sequence — the MLA analogue of GQA's
    `DenseBackend`, so DeepSeek's forward_dense is a thin wrapper over it."""

    def __init__(self, w: dict, *, num_heads: int, qk_nope: int, qk_rope: int,
                 v_head_dim: int, kv_lora_rank: int, q_lora_rank, rms_eps: float,
                 softmax_scale: float, layer_prefix=None):
        self.w = w
        self.num_heads = num_heads
        self.qk_nope = qk_nope
        self.qk_rope = qk_rope
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rms_eps = rms_eps
        self.softmax_scale = softmax_scale
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        return []                                       # dense: no paged cache

    def _write_kv(self, layer_idx, k, v, ctx) -> None:
        return None                                     # dense: nothing cached

    def _attn(self, layer_idx, q, k, v, ctx):
        """q/k: (T, nh, qk_nope+qk_rope), v: (T, nh, v_head_dim). Full causal
        softmax (fp32), single sequence (forward_dense)."""
        return dense_causal_attention(
            q, k, v, ctx.cu_seqlens_q, self.softmax_scale)
