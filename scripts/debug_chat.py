import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
text=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}], add_generation_prompt=True, tokenize=False)
print("TEXT:", repr(text)[:160])
ids=tok(text, add_special_tokens=False).input_ids
dev=npu_device(0)
m=DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
n0=len(ids)
for _ in range(24):
    t=torch.tensor(ids,dtype=torch.long,device=dev); pos=torch.arange(len(ids),device=dev)
    nxt=int(m.forward_dense(t,pos)[-1].float().argmax()); ids.append(nxt)
    if nxt==tok.eos_token_id: break
print("CHAT ANSWER:", repr(tok.decode(ids[n0:])))
