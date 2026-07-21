"""Verify exact-shape prefill/mixed ACL graphs against the eager FIA-v2 path."""
import gc
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import (
    CacheConfig, EngineConfig, ExecutionConfig, ModelConfig, SchedulerConfig)
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.entrypoints.llm import LLM


def _config(path, force_eager):
    return EngineConfig(
        model=ModelConfig(path, max_model_len=2048, dtype="bfloat16"),
        cache=CacheConfig(block_size=16, num_blocks=256),
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=8192),
        execution=ExecutionConfig(
            mode="graph", device_index=0, max_gear=32,
            force_eager=force_eager))


def _run(path, prompts, force_eager):
    llm = LLM(_config(path, force_eager))
    single = llm.generate([list(prompts[0])], max_tokens=64)
    pair = llm.generate([list(prompts[1]), list(prompts[2])], max_tokens=16)

    first = Request("mixed-a", list(prompts[1]), SamplingParams(max_tokens=8))
    second = Request("mixed-b", list(prompts[2]), SamplingParams(max_tokens=8))
    llm.engine.add_request(first)
    llm.engine.step()  # first request prefill; it is now a running decode request
    llm.engine.add_request(second)
    runner_stats = llm.engine.executor.runner.stats
    before_mixed = dict(runner_stats)
    llm.engine.step()  # late request must join the active decode batch immediately
    assert len(first.output_token_ids) == 2
    assert len(second.output_token_ids) == 1
    if force_eager:
        assert runner_stats["eager_steps"] == before_mixed["eager_steps"] + 1
    else:
        assert (runner_stats["prefill_graph_steps"]
                == before_mixed["prefill_graph_steps"] + 1)
        assert runner_stats["eager_steps"] == before_mixed["eager_steps"]
        assert (runner_stats["prefill_graph_capture_failures"]
                == before_mixed["prefill_graph_capture_failures"])
        assert (runner_stats["prefill_graph_capture_attempts"]
                == before_mixed["prefill_graph_capture_attempts"])
        assert (runner_stats["prefill_graph_online_captures"]
                == before_mixed["prefill_graph_online_captures"] == 0)
    while llm.engine.has_unfinished():
        llm.engine.step()

    oversized = llm.generate([list(prompts[0]) for _ in range(16)], max_tokens=4)
    for _ in range(50):
        llm.generate([list(prompts[0])], max_tokens=1)
    torch.npu.synchronize()
    result = {
        "single": single,
        "pair": pair,
        "mixed": [list(first.output_token_ids), list(second.output_token_ids)],
        "oversized": oversized,
    }
    stats = dict(llm.engine.executor.runner.stats)
    llm.close()
    del llm
    gc.collect()
    torch.npu.empty_cache()
    return result, stats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    prompts = [
        tokenizer("Explain how a transformer decodes text.").input_ids,
        tokenizer("The capital of France is").input_ids,
        tokenizer("Water is made of").input_ids,
    ]

    graph, stats = _run(path, prompts, force_eager=False)
    eager, eager_stats = _run(path, prompts, force_eager=True)

    assert graph == eager
    assert stats["prefill_graph_steps"] >= 53
    assert stats["prefill_graph_fallbacks"] > 0
    assert stats["eager_steps"] > 0
    assert eager_stats["prefill_graph_steps"] == 0
    print("graph_stats", stats)
    print("eager_stats", eager_stats)
    print("PREFILL GRAPH PASS")


if __name__ == "__main__":
    main()
