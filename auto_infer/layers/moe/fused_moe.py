"""Fused MoE (grouped-GEMM) for DeepSeek/Qwen3-MoE on Ascend NPU.

Replaces the naive per-expert Python loop with the CANN fused-MoE op chain: permute
tokens by expert once, run ONE grouped GEMM across all experts, unpermute +
weighted-combine — instead of a matmul per (expert, top-k slot).

Flow:
  npu_moe_init_routing         : sort/permute tokens so same-expert tokens are contiguous
  npu_moe_compute_expert_tokens: per-expert token counts (grouped-GEMM group_list)
  npu_grouped_matmul (gate_up) : one batched GEMM over all experts  [split_item=3]
  npu_swiglu                   : silu(gate)*up on the fused gate|up output
  npu_grouped_matmul (down)    : one batched GEMM over all experts
  npu_moe_finalize_routing     : unpermute + apply top-k weights + sum over top-k

Weights are pre-stacked into grouped-GEMM layout (build_expert_weights): w13 =
cat([gate.T, up.T]) per expert → (E, hidden, 2*inter); w2 = down.T → (E, inter, hidden).
"""
import torch


def build_expert_weights(w, prefix, n_experts, lo=0, hi=None):
    """Stack per-expert gate/up/down into grouped-GEMM layout for experts [lo, hi).
    Stored weights are torch-Linear (out, in); grouped_matmul does x @ w, so we
    transpose. Returns (w13 (E, hidden, 2*inter), w2 (E, inter, hidden))."""
    hi = n_experts if hi is None else hi
    g = [w[f"{prefix}experts.{e}.gate_proj.weight"] for e in range(lo, hi)]
    u = [w[f"{prefix}experts.{e}.up_proj.weight"] for e in range(lo, hi)]
    d = [w[f"{prefix}experts.{e}.down_proj.weight"] for e in range(lo, hi)]
    w13 = torch.stack([torch.cat([gi.t(), ui.t()], dim=1) for gi, ui in zip(g, u)])
    w2 = torch.stack([di.t() for di in d])
    return w13.contiguous(), w2.contiguous()


def fused_local_experts(x, expert_tokens, w13, w2):
    """Compute tokens already sorted by local expert by fused EP dispatch."""
    if expert_tokens.dtype is not torch.int64 or expert_tokens.ndim != 1:
        raise ValueError("local expert counts must be a one-dimensional int64 tensor")
    if w13.ndim != 3 or w2.ndim != 3 or expert_tokens.shape[0] != w13.shape[0]:
        raise ValueError("local expert counts and stacked weights must align")
    if w2.shape[0] != w13.shape[0] or x.shape[-1] != w13.shape[1]:
        raise ValueError("local grouped-GEMM weight shapes do not align")
    import torch_npu
    gate_up = torch_npu.npu_grouped_matmul(
        [x], [w13], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=1,
        output_dtype=x.dtype)[0]
    inter = torch_npu.npu_swiglu(gate_up)
    return torch_npu.npu_grouped_matmul(
        [inter], [w2], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=1,
        output_dtype=x.dtype)[0]


def fused_experts(x, topk_ids, topk_weights, w13, w2, n_experts):
    """Routed-expert compute via grouped GEMM. x: (T, hidden); topk_ids/weights:
    (T, top_k). Returns (T, hidden) = sum over top-k of weight * expert(x)."""
    import torch_npu
    num_tokens = x.shape[0]
    topk_ids = topk_ids.to(torch.int32)
    row_idx = (torch.arange(topk_ids.numel(), device=x.device, dtype=torch.int32)
               .view(-1, num_tokens).transpose(0, 1).contiguous())
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(x, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, n_experts).to(torch.int64)
    gate_up = torch_npu.npu_grouped_matmul(
        [sorted_tokens], [w13], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=0)[0]
    inter = torch_npu.npu_swiglu(gate_up)
    down = torch_npu.npu_grouped_matmul(
        [inter], [w2], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=0)[0]
    return torch_npu.npu_moe_finalize_routing(
        down, None, None, None, topk_weights.to(x.dtype),
        expanded_src_to_dst_row, topk_ids)
