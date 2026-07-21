"""DeepSeek-V2-Lite decode throughput: eager (PagedNpuExecutor, MLA+MoE via
forward(ctx)) vs SP6's ACL-graph decode (GraphPagedNpuExecutor, GraphMlaBackend
+ MoE captured into the same graph). Mirrors `bench_graph_decode.py`'s
eager-vs-graph structure (Qwen2) for DeepSeek's MLA+MoE forward.

  python scripts/bench_deepseek_decode.py /data1/models/DeepSeek-V2-Lite-Chat
"""
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor
from auto_infer.worker.model_runner import PagedNpuExecutor

BLOCK, NB, N = 16, 4096, 96
BATCHES = [1, 4, 16]
REPS = 3


def _cfg(path):
    return EngineConfig(model=ModelConfig(model_path=path),
                       cache=CacheConfig(block_size=BLOCK, num_blocks=NB),
                       scheduler=SchedulerConfig(max_num_batched_tokens=8192))


def bench(llm, pid, batch):
    reqs = [list(pid) for _ in range(batch)]
    llm.generate([list(pid) for _ in range(min(batch, 2))], max_tokens=4)  # warmup/capture
    torch.npu.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        outs = llm.generate([list(r) for r in reqs], max_tokens=N)
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(len(o) for o in outs) / min(ts)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    pid = tok("Explain how a transformer decodes text.", return_tensors="pt").input_ids[0].tolist()
    print(f"model=DeepSeek-V2-Lite  max_tokens={N}  reps={REPS}")
    print(f"{'batch':>6} {'eager tok/s':>12} {'graph tok/s':>12} {'speedup':>9}")
    for b in BATCHES:
        e = bench(LLM(_cfg(path), executor=PagedNpuExecutor(path, NB, BLOCK)), pid, b)
        torch.npu.empty_cache()
        g = bench(LLM(_cfg(path), executor=GraphPagedNpuExecutor(path, NB, BLOCK, max_gear=16)), pid, b)
        torch.npu.empty_cache()
        print(f"{b:>6} {e:>12.1f} {g:>12.1f} {g / e:>8.2f}x")


if __name__ == "__main__":
    main()
