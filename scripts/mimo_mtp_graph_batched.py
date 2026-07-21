"""End-to-end graph MTP spec-decode through EngineCore + scheduler (gear-captured
two-stage target/drafter pipeline). Multi-request LLM.generate with GraphMtpPagedNpuExecutor vs
plain greedy — per-request token-identity gate + throughput."""
import sys
import time

import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ModelConfig, SchedulerConfig,
                               SpecDecodeConfig)
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor
from auto_infer.worker.graph_mtp_runner import GraphMtpPagedNpuExecutor
from auto_infer.worker.model_runner import PagedNpuExecutor

path = "/data1/models/MiMo-7B-Base"
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
TEXTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "Once upon a time, in a small village at the edge of a forest,",
]
prompts = [tok(t).input_ids for t in TEXTS]
NB, BSZ = 256, 128


def _cfg(spec):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BSZ, num_blocks=NB),
        scheduler=SchedulerConfig(max_num_batched_tokens=8192,
                                  enable_chunked_prefill=True, enable_prefix_caching=True,
                                  long_prefill_token_threshold=8),
        spec_decode=spec)


def run(spec, executor):
    llm = LLM(_cfg(spec), executor=executor)
    t0 = time.time()
    outs = llm.generate([list(p) for p in prompts], max_tokens=96, eos_token_id=tok.eos_token_id)
    elapsed = time.time() - t0
    stats = dict(getattr(getattr(executor, "runner", None), "stats", {}))
    llm.close()
    return outs, elapsed, stats


base, tb, _ = run(None, PagedNpuExecutor(
    path, NB, BSZ, device_index=dev, max_num_batched_tokens=8192))
gbase, tg, _ = run(None, GraphPagedNpuExecutor(
    path, NB, BSZ, device_index=dev, max_gear=16))
sp, ts, mtp_stats = run(
    SpecDecodeConfig(depth), GraphMtpPagedNpuExecutor(
        path, NB, BSZ, device_index=dev,
        num_speculative_tokens=depth))
print(f"\ngraph-MTP vs plain(eager): per-req {[b == s for b, s in zip(base, sp)]}")
print(f"graph-MTP vs graph-plain (MATCHED numeric regime): "
      f"ALL={gbase == sp}  per-req {[b == s for b, s in zip(gbase, sp)]}")
print(f"  K={depth} plain {tb:.2f}s ({sum(len(o) for o in base) / tb:.1f} tok/s) | "
      f"graph-plain {tg:.2f}s ({sum(len(o) for o in gbase) / tg:.1f} tok/s) | "
      f"graph-MTP {ts:.2f}s ({sum(len(o) for o in sp) / ts:.1f} tok/s)  "
      f"-> {tg / ts:.2f}x vs matched graph-plain")
if mtp_stats.get("spec_steps"):
    print("  acceptance-by-position " + str([
        round(value / mtp_stats["spec_steps"], 4)
        for value in mtp_stats["accepted_per_position"]]))
if gbase != sp:
    raise SystemExit("graph-MTP token stream differs from matched graph-plain")
if not all(len(output) == 96 for output in sp):
    raise SystemExit("MTP output length gate failed")
if mtp_stats.get("graph", 0) <= 0:
    raise SystemExit("MTP graph path was not exercised")
