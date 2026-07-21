"""Qwen2 with custom Triton-Ascend SwiGLU kernel == default (seam integrated into model)."""
import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.qwen2 import Qwen2Model
from auto_infer.platform import npu_device
dev=npu_device(0); path="/data0/models/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(path)
ids=tok("The capital of France is").input_ids
def gen(use_kernel):
    m=Qwen2Model.from_pretrained(path, dev, torch.bfloat16)
    m.use_custom_kernels=use_kernel
    seq=ids[:]
    for _ in range(12):
        t=torch.tensor(seq,device=dev); pos=torch.arange(len(seq),device=dev)
        seq.append(int(m.forward_dense(t,pos)[-1].argmax()))
    return tok.decode(seq)
d=gen(False); k=gen(True)
print("default:", repr(d))
print("kernel :", repr(k))
print("MATCH" if d==k else "MISMATCH")
