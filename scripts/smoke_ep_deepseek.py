"""EP=2 DeepSeek-V2 MoE greedy (torchrun --nproc_per_node=2). Output must == single-card."""
import os, torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.distributed.parallel_state import init_distributed, tp_rank, tp_size
from auto_infer.models.deepseek_v2 import DeepseekV2Model

init_distributed()
local = int(os.environ.get("LOCAL_RANK", "0"))
dev = torch.device(f"npu:{local}")
path = "/data1/models/DeepSeek-V2-Lite-Chat"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
m = DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
ids = [tok.bos_token_id] + tok("The capital of France is").input_ids
for _ in range(8):
    t = torch.tensor(ids, dtype=torch.long, device=dev); pos = torch.arange(len(ids), device=dev)
    ids.append(int(m.forward_dense(t, pos)[-1].argmax()))
if tp_rank() == 0:
    print(f"EP{tp_size()} greedy:", repr(tok.decode(ids[1:])))
