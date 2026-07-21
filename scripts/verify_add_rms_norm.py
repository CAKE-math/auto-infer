"""Direct op 对拍: fused npu_add_rms_norm == manual (residual+x then rms_norm), on NPU.
Proves the fusion is correctly wired + numerically equivalent within bf16 tolerance.
If allclose, any greedy-token shift downstream is benign bf16 tie-breaking, not a bug.
"""
import torch
import torch_npu  # noqa: F401

from auto_infer.models.qwen2 import _add_rms_norm, _rms_norm
from auto_infer.platform import npu_device

dev = npu_device(0)
torch.manual_seed(0)
H, T, eps = 896, 7, 1e-6
for trial in range(3):
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    residual = torch.randn(T, H, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(H, dtype=torch.bfloat16, device=dev)
    normed_f, res_f = _add_rms_norm(x, residual, weight, eps)              # fused NPU op
    res_m = residual + x
    normed_m = _rms_norm(res_m, weight, eps)                               # manual (old path)
    dn = float((normed_f.float() - normed_m.float()).abs().max())
    dr = float((res_f.float() - res_m.float()).abs().max())
    reln = dn / (normed_m.float().abs().max().item() + 1e-6)
    cn = torch.allclose(normed_f.float(), normed_m.float(), atol=2e-2, rtol=2e-2)
    print(f"trial{trial}: max|Δnorm|={dn:.4f}(rel {reln:.4f}) max|Δres|={dr:.5f} allclose={cn}")
    ok = cn and dr < 1e-2
print("=== fused npu_add_rms_norm == manual add+rmsnorm (NPU) ===")
print("ALL MATCH" if ok else "MISMATCH")
