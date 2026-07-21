"""NPU verification for captured resident-dtype lm-head and greedy argmax."""
import hashlib
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import (
    CacheConfig, EngineConfig, ExecutionConfig, ModelConfig, SchedulerConfig)
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.entrypoints.llm import LLM


def _digest(rows):
    payload = ",".join(str(token) for row in rows for token in row).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _config(path):
    return EngineConfig(
        model=ModelConfig(path, max_model_len=2048, dtype="bfloat16"),
        cache=CacheConfig(block_size=16, num_blocks=4096),
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=8192),
        execution=ExecutionConfig(mode="graph", device_index=0, max_gear=32))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt = tokenizer("Explain how a transformer decodes text.").input_ids
    llm = LLM(_config(path))

    expected = None
    for batch in (1, 16, 4, 16):
        outputs = llm.generate([list(prompt) for _ in range(batch)], max_tokens=32)
        assert all(len(row) == 32 for row in outputs)
        if batch == 16 and expected is None:
            expected = _digest(outputs)
        elif batch == 16:
            assert _digest(outputs) == expected

    long_a = llm.generate([list(prompt)], max_tokens=256)
    long_b = llm.generate([list(prompt)], max_tokens=256)
    assert long_a == long_b and len(long_a[0]) == 256

    # A non-greedy request must consume captured logits via the external sampler.
    request = Request("sampled", list(prompt), SamplingParams(
        max_tokens=8, temperature=0.8, top_k=8, seed=7))
    llm.engine.add_request(request)
    while llm.engine.has_unfinished():
        llm.engine.step()

    torch.npu.synchronize()
    stats = llm.engine.executor.runner.stats
    assert stats["captured_greedy_steps"] >= 500
    assert stats["external_sampler_steps"] > 0
    print("long_digest", _digest(long_a))
    print("stats", stats)
    llm.close()
    print("GREEDY EPILOGUE GRAPH PASS")


if __name__ == "__main__":
    main()
