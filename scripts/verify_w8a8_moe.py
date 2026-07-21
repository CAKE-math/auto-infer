"""W8A8 fused MoE 对拍 (spec §8): int8 grouped-GEMM MoE vs bf16 fused MoE on NPU
(DeepSeek-V2-Lite, self-quantized experts — V2-Lite is bf16). Like the Qwen2 W8A8
check: quantized output should be ~bf16 (small per-token/per-channel quant error) and
keep coherent greedy tokens. Verifies the int8 grouped-GEMM path + scales."""
import torch
import torch_npu  # noqa: F401
from auto_infer.layers.moe.fused_moe import build_expert_weights, fused_experts
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device


def _quant_per_channel(weight):
    scale = weight.abs().amax(dim=1).clamp(min=1e-6) / 127.0
    quantized = (weight / scale.unsqueeze(1)).round().clamp(
        -127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.bfloat16).contiguous()


def _build_expert_weights_w8a8(weights, prefix, n_experts):
    w13, w2 = build_expert_weights(weights, prefix, n_experts)
    w13_i8, w13_scale = _quant_per_channel(w13)
    w2_i8, w2_scale = _quant_per_channel(w2)
    return w13_i8, w13_scale, w2_i8, w2_scale


def _fused_experts_w8a8(x, topk_ids, topk_weights, w13_i8, w13_scale,
                         w2_i8, w2_scale, n_experts):
    num_tokens = x.shape[0]
    topk_ids = topk_ids.to(torch.int32)
    row_idx = torch.arange(
        topk_ids.numel(), device=x.device, dtype=torch.int32
    ).view(-1, num_tokens).transpose(0, 1).contiguous()
    sorted_tokens, src2dst, expert_idx = torch_npu.npu_moe_init_routing(
        x, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expert_idx, n_experts).to(torch.int64)
    xq, x_scale = torch_npu.npu_dynamic_quant(sorted_tokens)
    gate_up = torch_npu.npu_grouped_matmul(
        [xq], [w13_i8], scale=[w13_scale], per_token_scale=[x_scale],
        group_list=expert_tokens, split_item=3, group_type=0,
        group_list_type=0, output_dtype=x.dtype)[0]
    intermediate = torch_npu.npu_swiglu(gate_up)
    iq, intermediate_scale = torch_npu.npu_dynamic_quant(intermediate)
    down = torch_npu.npu_grouped_matmul(
        [iq], [w2_i8], scale=[w2_scale],
        per_token_scale=[intermediate_scale], group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=0,
        output_dtype=x.dtype)[0]
    return torch_npu.npu_moe_finalize_routing(
        down, None, None, None, topk_weights.to(x.dtype), src2dst, topk_ids)

dev = npu_device(0); path = "/data1/models/DeepSeek-V2-Lite-Chat"
m = DeepseekV2Model.from_pretrained(path, dev, torch.bfloat16)
cfg = m.cfg
i = cfg.first_k_dense; p = f"model.layers.{i}.mlp."
w = m.w
torch.manual_seed(0)
T = 8
x = torch.randn(T, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
router = (x @ w[p + "gate.weight"].t()).float()
topk_w, topk_i = m.moe._gate(router, p)
topk_w = (topk_w * cfg.routed_scale).to(torch.bfloat16)

w13, w2 = build_expert_weights(w, p, cfg.n_routed)
bf16_out = fused_experts(x, topk_i, topk_w, w13, w2, cfg.n_routed).float()

w13i, w13s, w2i, w2s = _build_expert_weights_w8a8(w, p, cfg.n_routed)
w8a8_out = _fused_experts_w8a8(
    x, topk_i, topk_w, w13i, w13s, w2i, w2s, cfg.n_routed).float()

d = float((bf16_out - w8a8_out).abs().max())
rel = d / (bf16_out.abs().max().item() + 1e-6)
close = torch.allclose(bf16_out, w8a8_out, atol=0.15, rtol=0.15)
cos = torch.nn.functional.cosine_similarity(bf16_out.flatten(), w8a8_out.flatten(), dim=0).item()
print(f"W8A8 MoE 对拍: max|Δ|={d:.4f} rel={rel:.4f} cos_sim={cos:.5f} allclose(0.15)={close}")
print("=== W8A8 fused MoE ≈ bf16 fused MoE ===")
print("OK" if cos > 0.99 else "MISMATCH")
