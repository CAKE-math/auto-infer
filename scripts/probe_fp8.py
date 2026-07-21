import torch, torch_npu
dev="npu:0"
print("e4m3 dtype:", hasattr(torch, "float8_e4m3fn"))
ops = [o for o in dir(torch_npu) if "quant" in o.lower() or "fp8" in o.lower() or "hifloat" in o.lower() or "mm" in o.lower()]
print("torch_npu quant/fp8 ops:", [o for o in ops][:20])
# try fp8 cast
try:
    x = torch.randn(4,8,dtype=torch.bfloat16,device=dev)
    xf = x.to(torch.float8_e4m3fn)
    print("fp8 cast OK", xf.dtype)
except Exception as e:
    print("fp8 cast FAIL:", str(e)[:120])
