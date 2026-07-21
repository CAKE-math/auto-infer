"""Matched MiMo MTP probe for auto-infer.

Keep this workload byte-for-byte aligned with probe_vllm_mtp.py so throughput
and greedy output digests are directly comparable across runtimes.
"""
import hashlib
import json
import statistics
import sys
import time

from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                               ModelConfig, SchedulerConfig, SpecDecodeConfig)
from auto_infer.entrypoints.llm import LLM


path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/MiMo-7B-Base"
device_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
framework = "auto-infer"
disable_mtp = __import__("os").environ.get("MTP_DISABLE", "0") == "1"
force_eager = __import__("os").environ.get("MTP_FORCE_EAGER", "0") == "1"
if force_eager and not disable_mtp:
    raise ValueError("MTP_FORCE_EAGER is only valid with MTP_DISABLE=1")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
base_prompts = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "Once upon a time, in a small village at the edge of a forest,",
]
batch_size = int(__import__("os").environ.get("MTP_BATCH_SIZE", "4"))
prompts = [base_prompts[index % len(base_prompts)] for index in range(batch_size)]
prompt_ids = [tokenizer(text).input_ids for text in prompts]

load_start = time.perf_counter()
llm = LLM(EngineConfig(
    model=ModelConfig(model_path=path, max_model_len=512, dtype="bfloat16"),
    cache=CacheConfig(block_size=128, num_blocks=256),
    scheduler=SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=8,
    ),
    execution=ExecutionConfig(
        mode="graph" if disable_mtp else "graph_mtp",
        device_index=device_index, max_gear=16, force_eager=force_eager),
    spec_decode=None if disable_mtp else SpecDecodeConfig(),
    log_stats=True,
    log_stats_interval_s=3600.0,
))
load_seconds = time.perf_counter() - load_start


def generate():
    # Leaving eos_token_id unset is auto-infer's equivalent of ignore_eos=True.
    return llm.generate([list(ids) for ids in prompt_ids], max_tokens=32)


try:
    generate()  # warm the exact batch gear and the same prompt-cache entries
    stat_logger = llm.engine.stat_logger
    stat_logger._reset_window()
    elapsed_samples = []
    outputs = None
    sample_count = int(__import__("os").environ.get("MTP_SAMPLES", "5"))
    for _ in range(sample_count):
        start = time.perf_counter()
        outputs = generate()
        elapsed_samples.append(time.perf_counter() - start)
    assert outputs is not None
    digest = hashlib.sha256(json.dumps(outputs).encode()).hexdigest()[:16]
    median_elapsed = statistics.median(elapsed_samples)
    mean_elapsed = statistics.mean(elapsed_samples)
    result = {
        "framework": framework,
        "mtp_enabled": not disable_mtp,
        "batch_size": batch_size,
        "load_seconds": load_seconds,
        "elapsed_samples_seconds": elapsed_samples,
        "median_elapsed_seconds": median_elapsed,
        "elapsed_cv_percent": (
            statistics.pstdev(elapsed_samples) / mean_elapsed * 100.0),
        "throughput_tokens_per_second": sum(map(len, outputs)) / median_elapsed,
        "output_digest": digest,
        "output_lengths": list(map(len, outputs)),
        "output_token_ids": outputs,
        "spec_acceptance_rate": (
            stat_logger._spec_accepted / stat_logger._spec_steps
            if stat_logger._spec_steps else 0.0),
        "tokens_per_step": (
            1.0 + stat_logger._spec_accepted / stat_logger._spec_steps
            if stat_logger._spec_steps else 1.0),
        "graph_stats": dict(getattr(
            getattr(llm.engine.executor, "runner", None), "stats", {})),
    }
    print("MTP_PROBE " + json.dumps(result, sort_keys=True))
finally:
    llm.close()
