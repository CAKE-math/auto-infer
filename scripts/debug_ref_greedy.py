import torch
import transformers.utils.import_utils as iu
if not hasattr(iu,"is_torch_fx_available"): iu.is_torch_fx_available=lambda *a,**k: False
import transformers.utils as tu
if not hasattr(tu,"is_torch_fx_available"): tu.is_torch_fx_available=iu.is_torch_fx_available
from transformers import AutoTokenizer, AutoModelForCausalLM
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
hf=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
ids=[tok.bos_token_id]+tok("The capital of France is").input_ids
for _ in range(10):
    nxt=int(hf(torch.tensor([ids])).logits[0,-1].argmax()); ids.append(nxt)
print("REF greedy ids:", ids[1:])
print("REF greedy:", repr(tok.decode(ids[1:])))
