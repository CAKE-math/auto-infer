import torch
import transformers.utils.import_utils as iu
if not hasattr(iu,"is_torch_fx_available"): iu.is_torch_fx_available=lambda *a,**k: False
import transformers.utils as tu
if not hasattr(tu,"is_torch_fx_available"): tu.is_torch_fx_available=iu.is_torch_fx_available
from transformers import AutoTokenizer, AutoModelForCausalLM
path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids=tok("The capital of France is", return_tensors="pt").input_ids
hf=AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.float32)
lg=hf(ids).logits[0,-1].float(); top=lg.topk(5)
print("REF top5:", [(int(i), tok.decode([int(i)])) for i in top.indices])
