import torch
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids=[tok.bos_token_id]+tok("The capital of France is").input_ids
m=DeepseekV2Model.from_pretrained(path, device=torch.device("cpu"), dtype=torch.float32)
for _ in range(10):
    t=torch.tensor(ids,dtype=torch.long); pos=torch.arange(len(ids))
    ids.append(int(m.forward_dense(t,pos)[-1].argmax()))
print("CPU-fp32 greedy:", repr(tok.decode(ids[1:])))
