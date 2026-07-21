import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print("bos_token_id:", tok.bos_token_id, "add_bos:", getattr(tok,"add_bos_token",None))
base=tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
ids=[tok.bos_token_id]+base
print("with BOS ids:", ids)
dev=npu_device(0)
m=DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
for label,seq in [("no-bos",base),("with-bos",ids)]:
    t=torch.tensor(seq,dtype=torch.long,device=dev); pos=torch.arange(len(seq),device=dev)
    top=m.forward_dense(t,pos)[-1].float().topk(5)
    print(label,"top5:", [(int(i),tok.decode([int(i)])) for i in top.indices])
