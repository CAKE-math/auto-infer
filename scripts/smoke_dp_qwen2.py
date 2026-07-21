"""Data Parallel: AI_TP=1, torchrun --nproc_per_node=2. Each rank = independent DP replica
processing a different prompt (model replicated, no cross-rank comm)."""
import os, torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.distributed.parallel_state import init_distributed, tp_size, dp_size
from auto_infer.models.qwen2 import Qwen2Model

def main():
    init_distributed()
    local = int(os.environ.get("LOCAL_RANK", "0")); rank = int(os.environ.get("RANK", "0"))
    dev = torch.device(f"npu:{local}")
    path = "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    m = Qwen2Model.from_pretrained(path, dev, torch.bfloat16)   # TP=1: replicated
    prompts = ["The capital of France is", "The largest ocean on Earth is the"]
    ids = tok(prompts[rank % 2]).input_ids
    for _ in range(8):
        t = torch.tensor(ids, device=dev); pos = torch.arange(len(ids), device=dev)
        ids.append(int(m.forward_dense(t, pos)[-1].argmax()))
    print(f"[DP tp={tp_size()} dp={dp_size()}] rank{rank}: {tok.decode(ids)!r}")

if __name__ == "__main__":
    main()
