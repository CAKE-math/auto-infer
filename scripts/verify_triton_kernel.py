import torch, torch_npu
from auto_infer.platform import npu_device
from auto_infer.layers.kernels.swiglu_triton import silu_mul
dev=npu_device(0); torch.manual_seed(0)
x=torch.randn(4096, dtype=torch.float16, device=dev)
y=torch.randn(4096, dtype=torch.float16, device=dev)
ref=(torch.nn.functional.silu(x.float())*y.float())
out=silu_mul(x,y).float()
rel=(out-ref).abs().mean()/ref.abs().mean()
print(f"Triton-Ascend silu_mul vs torch rel-err={rel.item():.5f} ->", "OK" if rel.item()<0.01 else "HIGH")
