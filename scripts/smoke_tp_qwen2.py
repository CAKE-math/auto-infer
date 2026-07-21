"""TP=2 Qwen-family greedy (torchrun --nproc_per_node=2). Output must equal TP=1."""
import json, os, sys, torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.distributed.parallel_state import init_distributed, tp_rank, tp_size
from auto_infer.models.registry import get_model_class

init_distributed()
local = int(os.environ.get("LOCAL_RANK", "0"))
dev = torch.device(f"npu:{local}")
path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
with open(os.path.join(path, "config.json")) as f:
    model_cls = get_model_class(json.load(f)["architectures"][0])
m = model_cls.from_pretrained(path, dev, torch.bfloat16,
                              tp_rank=tp_rank(), tp_size=tp_size())
ids = tok("The capital of France is").input_ids
for _ in range(16):
    t = torch.tensor(ids, device=dev); pos = torch.arange(len(ids), device=dev)
    ids.append(int(m.forward_dense(t, pos)[-1].argmax()))
if tp_rank() == 0:
    print(f"TP{tp_size()} greedy:", repr(tok.decode(ids)))
