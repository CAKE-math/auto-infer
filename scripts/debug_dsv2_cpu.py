import torch
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids=tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
m=DeepseekV2Model.from_pretrained(path, device=torch.device("cpu"), dtype=torch.float32)
t=torch.tensor(ids,dtype=torch.long); pos=torch.arange(len(ids))
lg=m.forward_dense(t,pos)[-1].float(); top=lg.topk(5)
print("CPU-fp32 OUR top5:", [(int(i), tok.decode([int(i)])) for i in top.indices])
