"""Fused SwiGLU op 对拍: npu_swiglu(cat[gate,up]) == silu(gate)*up on NPU."""
import torch
import torch_npu  # noqa: F401
from auto_infer.platform import npu_device

dev = npu_device(0); torch.manual_seed(0)
ok = True
for T, D in [(7, 4864), (1, 4864), (32, 4864)]:
    gate = torch.randn(T, D, dtype=torch.bfloat16, device=dev)
    up = torch.randn(T, D, dtype=torch.bfloat16, device=dev)
    fused = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1))
    manual = torch.nn.functional.silu(gate) * up
    d = float((fused.float() - manual.float()).abs().max())
    rel = d / (manual.float().abs().max().item() + 1e-6)
    close = torch.allclose(fused.float(), manual.float(), atol=2e-2, rtol=2e-2)
    ok = ok and close
    print(f"T={T}: max|Δ|={d:.4f} rel={rel:.4f} allclose={close}")
print("=== fused npu_swiglu == silu*up ===")
print("ALL MATCH" if ok else "MISMATCH")
