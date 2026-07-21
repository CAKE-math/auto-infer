"""Measure the batch-queue async-scheduling overlap: async ON vs OFF on the SAME
executor, so the only difference is whether host scheduling/build/dispatch of
step N+1 overlaps the D2H/output-thread collect of step N.

SP4: the D2H (`.tolist()`) is no longer inline on the engine thread when async
is on — it's handed to a dedicated single-worker output thread
(`collect_async` -> Future, blocked on via `collect_result` only when the
in-flight queue is full). This is the async-must-win gate: async should now
BEAT sync (previously it was ~breakeven/regressed because both submit's
dispatch AND collect's D2H ran inline on the engine thread either way).

Async scheduling hides HOST latency behind device compute, so its relative win is
  host_overhead / (host_overhead + device_time)
— largest when device steps are cheap (small model / decode) and shrinks toward 0
as the model grows compute-bound. We sweep batch size, AND both executors
(eager PagedNpuExecutor and graph-replay GraphPagedNpuExecutor — the graph
path is where the win should compound most, since its per-step dispatch is
cheapest and the D2H/host is the exposed cost) to show the trend.

  python scripts/bench_async_scheduling.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import statistics
import os
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 8192
MAX_TOKENS = 128
BATCHES = [int(value) for value in os.getenv("AI_BENCH_BATCHES", "1,16").split(",")]
DEPTHS = [int(value) for value in os.getenv("AI_BENCH_DEPTHS", "0,1,2,3").split(",")]
REPS = int(os.getenv("AI_BENCH_REPS", "3"))

def build(path, depth):
    cfg = EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=8192),
        async_scheduling=depth > 0,
        async_batches=max(1, depth),
    )
    return LLM(cfg, executor=GraphPagedNpuExecutor(
        path, NUM_BLOCKS, BLOCK_SIZE, max_gear=64))


def bench(llm, prompt_ids, batch, max_tokens):
    reqs = [list(prompt_ids) for _ in range(batch)]
    llm.generate([list(prompt_ids) for _ in range(min(batch, 4))], max_tokens=8)  # warmup
    torch.npu.synchronize()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        outs = llm.generate([list(r) for r in reqs], max_tokens=max_tokens)
        torch.npu.synchronize()
        times.append(time.perf_counter() - t0)
    gen = sum(len(o) for o in outs)
    median = statistics.median(times)
    return median, gen / median, times


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    prompt = "Explain in detail how a large language model generates text, step by step."
    pid = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"model={path.split('/')[-1]}  prompt_len={len(pid)}  max_tokens={MAX_TOKENS}  reps={REPS}")
    print("\n=== executor=graph; depth=0 is synchronous ===")
    for batch in BATCHES:
        baseline = None
        for depth in DEPTHS:
            llm = build(path, depth)
            elapsed, tps, samples = bench(llm, pid, batch, MAX_TOKENS)
            llm.close()
            torch.npu.empty_cache()
            if baseline is None:
                baseline = tps
            print(f"batch={batch} depth={depth} median_s={elapsed:.6f} "
                  f"tok_s={tps:.3f} speedup={tps / baseline:.4f} samples={samples}")


if __name__ == "__main__":
    main()
