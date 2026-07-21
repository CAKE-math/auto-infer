import torch, torch_npu
from auto_infer.layers.quantization.w8a8 import quantize_weight, w8a8_linear
from auto_infer.platform import npu_device
dev=npu_device(0)
torch.manual_seed(0)
m,k,n = 32, 896, 4864
x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
W = torch.randn(n, k, dtype=torch.bfloat16, device=dev) * 0.1
ref = (x @ W.t()).float()
wi, ws = quantize_weight(W)
out = w8a8_linear(x, wi, ws).float()
rel = (out-ref).abs().mean() / ref.abs().mean()
print(f"W8A8 vs bf16 rel-err={rel.item():.4f} ->", "OK" if rel.item()<0.05 else "HIGH")
