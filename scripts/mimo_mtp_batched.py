"""Batched multi-request MTP spec-decode through the full EngineCore/scheduler.

Runs several prompts CONCURRENTLY via LLM.generate with SpecDecodeConfig(
proposer='mtp') on MiMo-7B, and compares to plain greedy (spec off). Greedy output
must be token-identical for EVERY request (rejection-sampler guarantee holds under
continuous batching), and spec should be faster wall-clock. MTP requires
non-chunked prefill (the head prefills each prompt when its main prefill completes).
"""
import sys
import time

import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ModelConfig, SchedulerConfig,
                               SpecDecodeConfig)
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

path = "/data1/models/MiMo-7B-Base"
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

TEXTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",   # dup -> exercises prefix cache
    "Once upon a time, in a small village at the edge of a great forest,",
]
prompts = [tok(t).input_ids for t in TEXTS]


def run(spec):
    cfg = EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=16, num_blocks=2048),
        scheduler=SchedulerConfig(max_num_batched_tokens=8192,
                                  enable_chunked_prefill=True, enable_prefix_caching=True,
                                  long_prefill_token_threshold=8),   # force chunking (8-tok chunks)
        spec_decode=spec)
    llm = LLM(cfg, executor=PagedNpuExecutor(path, 2048, 16, device_index=dev,
                                             max_num_batched_tokens=8192,
                                             num_speculative_tokens=depth))
    t0 = time.time()
    outs = llm.generate([list(p) for p in prompts], max_tokens=96,
                        eos_token_id=tok.eos_token_id)
    return outs, time.time() - t0


base, tb = run(None)
sp, ts = run(SpecDecodeConfig(depth))
ntok = sum(len(o) for o in sp)
print(f"\ngreedy MATCH (all {len(prompts)} reqs): {base == sp}")
print(f"  per-req match: {[b == s for b, s in zip(base, sp)]}")
for i, (b, s) in enumerate(zip(base, sp)):
    if b != s:
        d = next((j for j in range(min(len(b), len(s))) if b[j] != s[j]), min(len(b), len(s)))
        print(f"  req{i} diverge@{d}: plain {b[max(0, d - 2):d + 2]} vs spec {s[max(0, d - 2):d + 2]}")
print(f"  K={depth} plain {tb:.2f}s ({sum(len(o) for o in base) / tb:.1f} tok/s) | "
      f"spec {ts:.2f}s ({ntok / ts:.1f} tok/s)  ->  {tb / ts:.2f}x")

# solo re-run of any mismatched prompt: isolates batching interaction vs ULP
for i, (b, s) in enumerate(zip(base, sp)):
    if b != s:
        saved = prompts[:]
        prompts[:] = [saved[i]]
        b1, _ = run(None)
        s1, _ = run(SpecDecodeConfig(depth))
        print(f"  req{i} SOLO match: {b1[0] == s1[0]}")
        prompts[:] = saved
