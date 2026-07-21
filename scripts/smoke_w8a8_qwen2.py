"""Full-model W8A8 Qwen2 greedy vs bf16."""
import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.qwen2 import Qwen2Model
from auto_infer.platform import npu_device
dev=npu_device(0); path="/data0/models/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(path)
def gen(m, n=16):
    ids=tok("The capital of France is").input_ids[:]
    for _ in range(n):
        t=torch.tensor(ids,device=dev); pos=torch.arange(len(ids),device=dev)
        ids.append(int(m.forward_dense(t,pos)[-1].argmax()))
    return tok.decode(ids)
print("bf16 :", repr(gen(Qwen2Model.from_pretrained(path, dev, torch.bfloat16))))
print("w8a8 :", repr(gen(Qwen2Model.from_pretrained(path, dev, torch.bfloat16, quantize="w8a8"))))
