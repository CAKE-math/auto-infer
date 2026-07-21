"""Dense SwiGLU MLP block (gate/up/down proj + fused CANN SwiGLU) — the FFN used
by dense layers, MoE shared experts, and the naive routed-expert path. Reads
weights by name from a weight dict. NPU-only — no CPU fallback."""
import torch


def _gate_up_projection(x, w, prefix):
    packed = prefix + "gate_up_proj.weight"
    if packed in w:
        return x @ w[packed].t()
    gate = x @ w[prefix + "gate_proj.weight"].t()
    up = x @ w[prefix + "up_proj.weight"].t()
    return torch.cat([gate, up], dim=-1)


def swiglu_mlp(x, w, prefix):
    import torch_npu
    gate_up = _gate_up_projection(x, w, prefix)
    inter = torch_npu.npu_swiglu(gate_up)     # fused CANN SwiGLU (§5)
    return inter @ w[prefix + "down_proj.weight"].t()
