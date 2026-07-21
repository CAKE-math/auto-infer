"""Smoke the observability StatLogger on the real paged NPU engine path:
run a batch of (partly shared-prefix) prompts through LLM.generate with
log_stats=True and watch the periodic [engine] metrics lines."""
import sys

import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

path = "/data1/models/Qwen3-0.6B"
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

cfg = EngineConfig(
    model=ModelConfig(model_path=path),
    cache=CacheConfig(block_size=16, num_blocks=2048),
    scheduler=SchedulerConfig(max_num_batched_tokens=4096),
    log_stats=True, log_stats_interval_s=0.5)
llm = LLM(cfg, executor=PagedNpuExecutor(path, 2048, 16, device_index=dev))

# 8 identical + 8 identical -> shared prefixes exercise the prefix-cache hit metric
p1 = tok("The capital of France is a city that").input_ids
p2 = tok("In a distant galaxy, a small robot learned to").input_ids
prompts = [list(p1) for _ in range(8)] + [list(p2) for _ in range(8)]
outs = llm.generate(prompts, max_tokens=64, eos_token_id=tok.eos_token_id)
print(f"done: {len(outs)} seqs, total gen tokens = {sum(len(o) for o in outs)}")
