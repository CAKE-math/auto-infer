"""W8A8 per-token dynamic quantization (spec sec 8).
weight: offline per-output-channel int8; activation: runtime per-token dynamic
int8; int8 GEMM via torch_npu.npu_quant_matmul. DeepSeek deployment precision."""
import torch


def quantize_weight(W: torch.Tensor):
    """W: (out, in) -> (w_int8_t (in,out), w_scale (out,)). Per-output-channel."""
    scale = W.abs().amax(dim=1, keepdim=True) / 127.0          # (out,1)
    w_int8 = (W / scale).round().clamp(-127, 127).to(torch.int8)
    return w_int8.t().contiguous(), scale.squeeze(1).float()


def w8a8_linear(x, w_int8_t, w_scale, bias=None):
    """x: (m, in) -> (m, out). Activation per-token dynamic quant + int8 GEMM."""
    import torch_npu
    x_int8, pertoken = torch_npu.npu_dynamic_quant(x)
    return torch_npu.npu_quant_matmul(x_int8, w_int8_t, w_scale,
                                      pertoken_scale=pertoken, bias=bias,
                                      output_dtype=x.dtype)
