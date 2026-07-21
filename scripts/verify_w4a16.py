import torch, torch_npu
from auto_infer.platform import npu_device
dev=npu_device(0); torch.manual_seed(0)
m,k,n,G = 32, 896, 4864, 128
x = torch.randn(m,k,dtype=torch.bfloat16,device=dev)
W = (torch.randn(n,k,dtype=torch.bfloat16,device=dev)*0.1)        # (out,in)
ref = (x @ W.t()).float()
# group-wise int4 along in-dim (k), groups of G
Wg = W.view(n, k//G, G)
scale = Wg.abs().amax(dim=2,keepdim=True)/7.0                     # (n, k//G, 1)
wq = (Wg/scale).round().clamp(-8,7).to(torch.int32).view(n,k)    # (out,in)
wq_t = wq.t().contiguous()                                       # (in,out)
packed = torch_npu.npu_convert_weight_to_int4pack(wq_t)
asc = scale.squeeze(2).t().contiguous().to(x.dtype)              # (k//G, n)
out = torch_npu.npu_weight_quant_batchmatmul(x, packed, antiquant_scale=asc, antiquant_group_size=G).float()
rel=(out-ref).abs().mean()/ref.abs().mean()
print(f"W4A16 group{G} vs bf16 rel-err={rel.item():.4f} ->", "OK" if rel.item()<0.06 else "HIGH")
