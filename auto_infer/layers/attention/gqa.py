"""GQA attention backends (Qwen2/Qwen3 family): paged-FIA (`GqaFIABackend`),
full-softmax dense bring-up (`DenseBackend`), ACL-graph decode
(`GraphGqaBackend`). All share `_GqaProjRopeMixin` (q/k/v proj + optional
QK-Norm + RoPE + o_proj); only the KV-write / core-attention op pair differs."""
import torch

from auto_infer.layers.attention.base import (
    AttentionBackend, _lin, dense_causal_attention)
from auto_infer.layers.attention.graph_fia import GraphFiaLifecycle
from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.layers.rotary_embedding import rotate_half as _rotate_half


def _split_qkv(projected: torch.Tensor, q_size: int, kv_size: int):
    """Return view-only Q/K/V slices from a packed projection result."""
    return projected.split((q_size, kv_size, kv_size), dim=-1)


class _GqaProjRopeMixin:
    """Shared GQA q/k/v-proj + RoPE + o_proj/tp_all_reduce wrapper — the part
    of Qwen2's attention body that's IDENTICAL across execution modes (eager
    paged FIA, dense full-softmax, ACL-graph FIA-v2); only the KV-write +
    core-attention op pair (`_write_kv`/`_attn`, implemented per subclass)
    differs. Mixed in before `AttentionBackend` so it supplies the concrete
    `attention()` the ABC requires."""

    def attention(self, layer_idx: int, x: torch.Tensor, ctx) -> torch.Tensor:
        from auto_infer.distributed.parallel_state import tp_all_reduce
        T = x.shape[0]
        n_q, n_kv, hd = self.n_q_heads, self.n_kv_heads, self.head_dim
        cos_h, sin_h = ctx.cos, ctx.sin
        w = self.w                                    # index by name (model owns naming)
        p = self.layer_prefix(layer_idx) + "self_attn."
        # bias is optional (Qwen2 has it; Qwen3 sets attention_bias=false).
        qkv = _lin(x, w[p + "qkv_proj.weight"], w.get(p + "qkv_proj.bias"))
        q, k, v = _split_qkv(qkv, n_q * hd, n_kv * hd)
        q = q.view(T, n_q, hd)
        k = k.view(T, n_kv, hd)
        v = v.view(T, n_kv, hd)
        qn = w.get(p + "q_norm.weight")               # Qwen3 QK-Norm (per-head RMSNorm on head_dim)
        if qn is not None:
            q = _rms_norm(q, qn, self.rms_eps)
            k = _rms_norm(k, w[p + "k_norm.weight"], self.rms_eps)
        q = q * cos_h + _rotate_half(q) * sin_h
        k = k * cos_h + _rotate_half(k) * sin_h

        self._write_kv(layer_idx, k, v, ctx)
        o = self._attn(layer_idx, q, k, v, ctx).reshape(T, n_q * hd)
        return tp_all_reduce(_lin(o, w[p + "o_proj.weight"]))


class GqaFIABackend(_GqaProjRopeMixin, AttentionBackend):
    """Eager paged-FIA path: canonical KV layout, plain FIA call. The default
    attention backend for the GQA family."""

    def __init__(self, n_q_heads: int, n_kv_heads: int, head_dim: int, scale: float,
                 num_layers: int, device, dtype, w: dict = None, layer_prefix=None, rms_eps=None):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.w = w                    # model weight dict (indexed by name in the mixin)
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self.rms_eps = rms_eps                        # for Qwen3 QK-Norm (None if no q_norm)

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        """Per-layer paged KV cache: (2, num_blocks, block_size, n_kv, head_dim).
        Uses the num_layers/device/dtype fixed at construction."""
        shape = (2, num_blocks, block_size, self.n_kv_heads, self.head_dim)
        return [torch.zeros(shape, device=self.device, dtype=self.dtype)
                for _ in range(self.num_layers)]

    def _write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, ctx) -> None:
        from auto_infer.layers.attention.base import write_kv
        key_cache, value_cache = ctx.kv_caches[layer_idx][0], ctx.kv_caches[layer_idx][1]
        write_kv(k, v, key_cache, value_cache, ctx.slot_mapping)

    def _attn(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ctx) -> torch.Tensor:
        from auto_infer.layers.attention.base import paged_fia
        key_cache, value_cache = ctx.kv_caches[layer_idx][0], ctx.kv_caches[layer_idx][1]
        # key_cache is (num_blocks, block_size, n_kv, head_dim) after the [0] index
        # dropped the leading (K,V) axis — read blocks/block_size off axes 0/1.
        num_blocks = key_cache.shape[0]
        block_size = key_cache.shape[1]
        k_fia = key_cache.view(num_blocks, block_size, self.n_kv_heads * self.head_dim)
        v_fia = value_cache.view(num_blocks, block_size, self.n_kv_heads * self.head_dim)
        return paged_fia(q, k_fia, v_fia, ctx.block_table, block_size=block_size,
                          actual_seq_q=ctx.cu_seqlens_q, actual_seq_kv=ctx.seqlens_kv,
                          num_kv_heads=self.n_kv_heads, num_heads=self.n_q_heads,
                          scale=self.scale, atten_mask=ctx.attn_mask)


class DenseBackend(_GqaProjRopeMixin, AttentionBackend):
    """Same proj/RoPE/o_proj wrapper as `GqaFIABackend`, but the core is a full
    causal softmax over the whole (T) sequence, fp32 accumulation, no paging/
    KV-reuse — bring-up / HF-parity path. `ctx.cu_seqlens_q` (cumulative
    per-request boundaries) splits a batch into independent causal segments so
    multi-request TND batches degrade to per-request full attention."""

    def __init__(self, n_q_heads: int, n_kv_heads: int, head_dim: int, scale: float,
                 w: dict = None, layer_prefix=None, rms_eps=None):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.w = w                    # model weight dict (indexed by name in the mixin)
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self.rms_eps = rms_eps                        # for Qwen3 QK-Norm (None if no q_norm)

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        return []

    def _write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, ctx) -> None:
        return None

    def _attn(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ctx) -> torch.Tensor:
        """Assumes q-len == kv-len (full causal self-attention over the whole
        sequence passed in this call) — valid for prefill / bring-up only.
        Must NOT be wired into an incremental decode step (q-len 1, kv-len >
        1), which this implementation does not handle."""
        n_q, n_kv = self.n_q_heads, self.n_kv_heads
        rep = n_q // n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        return dense_causal_attention(
            q, k, v, ctx.cu_seqlens_q, self.scale)


class GraphGqaBackend(_GqaProjRopeMixin, GraphFiaLifecycle, AttentionBackend):
    """ACL-graph-capturable paged attention: same proj/RoPE/o_proj wrapper as
    `GqaFIABackend`, but KV is NZ-layout paged KV written via
    `npu_scatter_pa_kv_cache`, and attention runs via FIA-v2 `.out` (the
    graph-capturable variant — no workspace sync, unlike plain FIA).

    Two sub-modes, toggled by `capturing`:
      * capturing=True (inside `torch.npu.graph(...)`): each layer's FIA call
        is wrapped in `graph_task_group_begin/end`; the resulting handle plus
        the exact (q, knz, vnz, o, lse) tensors the graph was captured against
        are appended to `self.reg`, one entry per layer in call order (layers
        run 0..num_layers-1 exactly once per capture, so list order == layer
        order — no explicit indexing needed).
      * capturing=False (eager / warmup): plain `.out` call, no group wrap.

    `update(ctx)` re-issues the SAME FIA call (same static q/knz/vnz/o/lse
    tensors, same block_table/mask/qlen_cum objects — those are the runner's
    static per-gear buffers, mutated in place before `update`) but with the
    step's fresh `ctx.seqlens_kv`, inside a `graph_task_update_begin/end`
    bracket — this is what lets one captured graph replay correctly across
    steps with different per-request KV lengths.

    `self.reg` is a single shared list; the runner swaps a per-gear reg list
    into `self.reg` before calling `update()` for that gear (each gear captures
    its own graph/handles against its own static buffers, but all gears share
    this one backend instance + the one NZ KV cache — see `_Gear`/`_capture`
    in `graph_decode_runner.py`).
    """

    NZ = 16  # FRACTAL_NZ inner-tile width for npu_scatter_pa_kv_cache / FIA-v2
    supports_prefill_graph = True

    def __init__(self, n_q_heads: int, n_kv_heads: int, head_dim: int, scale: float,
                 num_layers: int, device, dtype, w: dict = None, layer_prefix=None, rms_eps=None):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.w = w                    # model weight dict (indexed by name in the mixin)
        self.layer_prefix = layer_prefix or (lambda i: f"model.layers.{i}.")
        self.rms_eps = rms_eps                        # for Qwen3 QK-Norm (None if no q_norm)
        self._init_graph_fia()

    @property
    def _fia_query_heads(self):
        return self.n_q_heads

    @property
    def _fia_kv_heads(self):
        return self.n_kv_heads

    @property
    def _fia_scale(self):
        return self.scale

    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        """Per-layer NZ KV cache: TWO plain tensors (k, v), each
        (num_blocks, block_size, n_kv_heads, head_dim). The NZ reinterpretation
        happens via `.view()` at write/attn time (contiguous-memory bitcast),
        not in the allocated shape itself."""
        shape = (num_blocks, block_size, self.n_kv_heads, self.head_dim)
        return [(torch.zeros(shape, device=self.device, dtype=self.dtype),
                 torch.zeros(shape, device=self.device, dtype=self.dtype))
                for _ in range(self.num_layers)]

    def _write_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, ctx) -> None:
        """`npu_scatter_pa_kv_cache` with NZ views, scatter by `ctx.slot_mapping`."""
        import torch_npu
        kc, vc = ctx.kv_caches[layer_idx]
        num_blocks, block_size = kc.shape[0], kc.shape[1]
        torch_npu.npu_scatter_pa_kv_cache(
            k.contiguous(), v.contiguous(),
            kc.view(num_blocks, self.n_kv_heads * self.head_dim // self.NZ, block_size, self.NZ),
            vc.view(num_blocks, self.n_kv_heads * self.head_dim // self.NZ, block_size, self.NZ),
            ctx.slot_mapping)

    def _attn(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              ctx) -> torch.Tensor:
        """FIA-v2 `.out`, capture-aware. Returns `o` (T, n_q_heads, head_dim) —
        matches `GqaFIABackend._attn`'s contract so
        `_GqaProjRopeMixin.attention`'s `.reshape(T, n_q*hd)` works unchanged."""
        kc, vc = ctx.kv_caches[layer_idx]
        block_size = kc.shape[1]
        knz = kc.view(-1, self.n_kv_heads, self.head_dim // self.NZ, block_size, self.NZ)
        vnz = vc.view(-1, self.n_kv_heads, self.head_dim // self.NZ, block_size, self.NZ)
        T = q.shape[0]
        o = torch.empty(T, self.n_q_heads, self.head_dim, dtype=q.dtype, device=q.device)
        lse = torch.empty(T, dtype=torch.float32, device=q.device)
        self._run_graph_fia(q, knz, vnz, o, lse, ctx, block_size)
        return o
