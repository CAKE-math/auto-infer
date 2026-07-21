"""Attention base: the execution-mode seam (`AttentionBackend` ABC), the shared
W8A8-aware `_lin`, and the paged-KV torch_npu ops (`write_kv`/`paged_fia`) that
both the GQA (gqa.py) and MLA (mla.py) backend families build on."""
from abc import ABC, abstractmethod

import torch


def _lin(x: torch.Tensor, W, bias=None) -> torch.Tensor:
    """W8A8-aware linear dispatch: `W` is either a plain `(out, in)` tensor or a
    W8A8 `(int8_weight_transposed, per_output_channel_scale)` tuple. Backends own
    the whole attention sub-block (incl. q/k/v/o projections), so this dispatch
    lives with them."""
    if isinstance(W, tuple):
        from auto_infer.layers.quantization.w8a8 import w8a8_linear
        return w8a8_linear(x, W[0], W[1], bias=bias)
    out = x @ W.t()
    return out + bias if bias is not None else out


class AttentionBackend(ABC):
    """Execution-mode seam: the backend owns the WHOLE attention sub-block for
    one layer, returning a tensor ready to add straight to the residual."""

    @abstractmethod
    def alloc_kv_caches(self, num_blocks: int, block_size: int) -> list:
        """Per-layer KV cache list (backend-specific shape/dtype/device)."""
        raise NotImplementedError

    @abstractmethod
    def attention(self, layer_idx: int, x: torch.Tensor, ctx) -> torch.Tensor:
        """x: normed hidden (T, hidden) for this layer. Backends index the model
        weight dict `self.w` by name via the model-owned `layer_prefix(layer_idx)`.
        ctx: the step's `ForwardContext` (carries `cos`/`sin` rope tables the
        model computed once, plus KV-cache/paging metadata). Returns the
        attention sublayer output — projected, o_proj'd, tp-reduced — (T,
        hidden), ready for the caller to add to the residual."""
        raise NotImplementedError


def write_kv(key, value, key_cache, value_cache, slot_mapping):
    """Scatter this step's K/V into the paged cache by slot index."""
    import torch_npu
    torch_npu._npu_reshape_and_cache(key=key, value=value, key_cache=key_cache,
                                     value_cache=value_cache, slot_indices=slot_mapping)


def paged_fia(query, key, value, block_table, *, block_size, actual_seq_q,
              actual_seq_kv, num_kv_heads, num_heads, scale, atten_mask=None,
              sparse_mode=3):
    """FIA over paged KV. query: (T, num_heads, hd); key/value: 3D cache views
    (num_blocks, block_size, num_kv_heads*hd). actual_seq_q cumulative, actual_seq_kv
    per-sequence. Returns attention output (T, num_heads, hd)."""
    import torch_npu
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query, key=key, value=value, block_table=block_table,
        input_layout="TND", block_size=block_size,
        actual_seq_lengths=actual_seq_q, actual_seq_lengths_kv=actual_seq_kv,
        num_key_value_heads=num_kv_heads, num_heads=num_heads, scale=scale,
        atten_mask=atten_mask, sparse_mode=sparse_mode)
    return out


def dense_causal_attention(query, key, value, cumulative_lengths, scale):
    """Full-softmax causal attention over independent packed TND segments."""
    outputs = []
    start = 0
    for end in cumulative_lengths:
        q = query[start:end].transpose(0, 1)
        k = key[start:end].transpose(0, 1)
        v = value[start:end].transpose(0, 1)
        length = end - start
        causal = torch.triu(torch.full(
            (length, length), float("-inf"), device=query.device,
            dtype=torch.float32), diagonal=1)
        scores = (q.float() @ k.float().transpose(-1, -2)) * scale + causal
        probability = torch.softmax(scores, dim=-1).to(query.dtype)
        outputs.append((probability @ v).transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)
