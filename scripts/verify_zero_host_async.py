#!/usr/bin/env python3
"""NPU correctness gates for zero-host-bubble graph decode."""
import hashlib
import json
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import (
    CacheConfig, EngineConfig, ExecutionConfig, ModelConfig, SchedulerConfig)
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.entrypoints.llm import LLM


def _digest(outputs):
    payload = json.dumps(outputs, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _build(model, asynchronous):
    return LLM(EngineConfig(
        model=ModelConfig(model, max_model_len=2048, dtype="bfloat16"),
        cache=CacheConfig(block_size=16, num_blocks=904),
        scheduler=SchedulerConfig(
            max_num_seqs=256, max_num_batched_tokens=8192),
        execution=ExecutionConfig(
            mode="graph", max_gear=32, max_prefill_tokens=256),
        async_scheduling=asynchronous,
        async_batches=2))


def _staggered(llm, prompt):
    arrivals = {
        0: [("a", 17)],
        2: [("b", 11), ("c", 7)],
        5: [("d", 5)],
    }
    results = {}
    step = 0
    while step < 256:
        for rid, maximum in arrivals.get(step, ()):
            llm.engine.add_request(Request(
                rid, list(prompt), SamplingParams(max_tokens=maximum)))
        if not llm.engine.has_unfinished() and step > max(arrivals):
            break
        for request in llm.engine.step():
            results[request.request_id] = list(request.output_token_ids)
        step += 1
    if set(results) != {"a", "b", "c", "d"}:
        raise RuntimeError(f"staggered run did not finish: {sorted(results)}")
    return [results[rid] for rid in ("a", "b", "c", "d")]


def _abort(llm, prompt):
    for rid in ("keep", "abort"):
        llm.engine.add_request(Request(
            rid, list(prompt), SamplingParams(max_tokens=12)))
    for _ in range(3):
        llm.engine.step()
    llm.engine.abort("abort")
    result = None
    while llm.engine.has_unfinished():
        for request in llm.engine.step():
            if request.request_id == "keep":
                result = list(request.output_token_ids)
    if result is None:
        raise RuntimeError("surviving request did not finish")
    return result


def _suite(model, prompt, asynchronous):
    llm = _build(model, asynchronous)
    try:
        b1 = llm.generate([list(prompt)], max_tokens=32)
        b16 = llm.generate([list(prompt) for _ in range(16)], max_tokens=32)
        staggered = _staggered(llm, prompt)
        eos = b1[0][0]
        eos_outputs = llm.generate(
            [list(prompt), list(prompt)], max_tokens=8, eos_token_id=eos)
        aborted = _abort(llm, prompt)
        return {
            "b1": b1,
            "b16": b16,
            "staggered": staggered,
            "eos": eos_outputs,
            "abort_survivor": aborted,
        }
    finally:
        llm.close()
        torch.npu.empty_cache()


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    prompt = AutoTokenizer.from_pretrained(model)(
        "Explain how a transformer decodes text.").input_ids
    sync = _suite(model, prompt, False)
    asynchronous = _suite(model, prompt, True)
    if asynchronous != sync:
        mismatches = [
            name for name in sync if sync[name] != asynchronous[name]]
        raise RuntimeError(f"sync/async mismatch: {mismatches}")
    print(json.dumps({
        "pass": True,
        "digest": _digest(sync),
        "cases": {
            name: [len(row) for row in value]
            if isinstance(value, list) and value and isinstance(value[0], list)
            else len(value)
            for name, value in sync.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
