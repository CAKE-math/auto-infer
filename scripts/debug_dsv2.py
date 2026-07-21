import torch, torch_npu
import transformers.utils.import_utils as iu
if not hasattr(iu, "is_torch_fx_available"): iu.is_torch_fx_available = lambda *a, **k: False
import transformers.utils as tu
if not hasattr(tu, "is_torch_fx_available"): tu.is_torch_fx_available = iu.is_torch_fx_available
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device

path="/data1/models/DeepSeek-V2-Lite-Chat"
tok=AutoTokenizer.from_pretrained(path, trust_remote_code=True)
ids=tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
print("input ids:", ids, [tok.decode([i]) for i in ids])
dev=npu_device(0)
m=DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
t=torch.tensor(ids,dtype=torch.long,device=dev); pos=torch.arange(len(ids),device=dev)
lg=m.forward_dense(t,pos)[-1].float()
top=lg.topk(5)
print("OUR top5:", [(int(i), tok.decode([int(i)])) for i in top.indices])
