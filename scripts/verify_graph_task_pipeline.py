"""NPU verification for side-stream graph-task updates.

Exercises first replay, changing KV lengths, and alternating captured gears.
"""
import hashlib
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import (
    CacheConfig,
    EngineConfig,
    ExecutionConfig,
    ModelConfig,
    SchedulerConfig,
)
from auto_infer.entrypoints.llm import LLM


def _digest(rows):
    payload = ",".join(str(token) for row in rows for token in row).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt = tokenizer("Explain how a transformer decodes text.").input_ids
    config = EngineConfig(
        model=ModelConfig(path, max_model_len=2048, dtype="bfloat16"),
        cache=CacheConfig(block_size=16, num_blocks=4096),
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=8192),
        execution=ExecutionConfig(mode="graph", device_index=0, max_gear=32),
    )
    llm = LLM(config)
    first_by_batch = {}
    for batch in (1, 16, 4, 16):
        outputs = llm.generate([list(prompt) for _ in range(batch)], max_tokens=24)
        assert all(len(row) == 24 for row in outputs)
        digest = _digest(outputs)
        if batch in first_by_batch:
            assert digest == first_by_batch[batch]
        first_by_batch[batch] = digest
        print(f"gear batch={batch} digest={digest}")

    long_a = llm.generate([list(prompt)], max_tokens=200)
    long_b = llm.generate([list(prompt)], max_tokens=200)
    assert long_a == long_b
    assert len(long_a[0]) == 200
    torch.npu.synchronize()
    assert llm.engine.executor.runner.stats["graph_steps"] >= 490
    print("long_replay_digest", _digest(long_a))
    print("stats", llm.engine.executor.runner.stats)
    llm.close()
    print("GRAPH TASK PIPELINE PASS")


if __name__ == "__main__":
    main()
