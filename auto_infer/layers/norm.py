"""Fused RMSNorm (torch_npu.npu_rms_norm). NPU-only — no CPU fallback."""
import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import torch_npu
    out, _ = torch_npu.npu_rms_norm(x, weight, eps)
    return out


def add_rms_norm(x, residual, weight, eps):
    """Fused residual-add + RMSNorm (spec §5; vLLM-ascend always-on fusion) —
    one kernel instead of add+norm. Returns (normed, new_residual = residual + x)."""
    import torch_npu
    normed, _, new_res = torch_npu.npu_add_rms_norm(x, residual, weight, eps)
    return normed, new_res
