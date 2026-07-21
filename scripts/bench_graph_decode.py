"""SP3 payoff: graph-replay decode vs eager decode latency. Same model/workload,
only the executor differs (GraphPagedNpuExecutor = ACL-graph decode vs
PagedNpuExecutor = eager forward(ctx) per step). Graph should collapse the
~30 ms/token Python-dispatch tax to ~device time.

  python scripts/bench_graph_decode.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 4096
N_GEN = 128
BATCHES = [1, 8, 32]
REPS = 3


def _cfg(path):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=8192))


def bench(llm, pid, batch):
    reqs = [list(pid) for _ in range(batch)]
    llm.generate([list(pid) for _ in range(min(batch, 4))], max_tokens=8)  # warmup/capture
    torch.npu.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        outs = llm.generate([list(r) for r in reqs], max_tokens=N_GEN)
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return sum(len(o) for o in outs) / min(ts)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    pid = tok("Explain how a transformer decodes text.", return_tensors="pt").input_ids[0].tolist()
    print(f"model={path.split('/')[-1]}  max_tokens={N_GEN}  reps={REPS}")
    print(f"{'batch':>6} {'eager tok/s':>12} {'graph tok/s':>12} {'speedup':>9}")
    for b in BATCHES:
        e = bench(LLM(_cfg(path), executor=PagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE)), pid, b)
        torch.npu.empty_cache()
        g = bench(LLM(_cfg(path), executor=GraphPagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE, max_gear=32)), pid, b)
        torch.npu.empty_cache()
        print(f"{b:>6} {e:>12.1f} {g:>12.1f} {g / e:>8.2f}x")


if __name__ == "__main__":
    main()
