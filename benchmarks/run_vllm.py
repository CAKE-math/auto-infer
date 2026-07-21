import os
import sys
import time

# Memory measurement initializes torch-npu in this process. Keep vLLM's engine
# in-process as well; forking after that initialization is rejected by torch-npu.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import torch_npu  # noqa: F401
from vllm import LLM, SamplingParams

from benchmarks.common import (
    emit_report, load_manifest, runtime_async_scheduling,
    runtime_kv_block_size, summarize, token_digest, validate_comparison_result)


def main():
    # omni-npu 0.14's patch loader inspects sys.argv[2] for the model path even
    # when the Python LLM API is used. Accept an optional compatibility slot.
    framework = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
    manifest = load_manifest(sys.argv[1])
    block_size = manifest["kv_block_size"]
    if manifest["usable_kv_tokens"] % block_size:
        raise ValueError("usable_kv_tokens must be divisible by kv_block_size")
    torch.npu.reset_peak_memory_stats()
    load_start = time.perf_counter()
    llm = LLM(model=manifest["model"], dtype=manifest["dtype"],
              trust_remote_code=True, max_model_len=manifest["max_model_len"],
              kv_cache_memory_bytes=manifest["kv_cache_memory_bytes"],
              enforce_eager=False,
              seed=manifest["seed"])
    prompt_ids = llm.get_tokenizer().encode(manifest["prompt"])
    load_seconds = time.perf_counter() - load_start
    runtime_block_size = runtime_kv_block_size(llm, fallback=block_size)
    if manifest["usable_kv_tokens"] % runtime_block_size:
        raise ValueError("usable_kv_tokens must be divisible by runtime block size")
    runtime_blocks = manifest["usable_kv_tokens"] // runtime_block_size

    def run(batch, output_tokens):
        params = SamplingParams(max_tokens=output_tokens,
                                temperature=manifest["temperature"], seed=manifest["seed"],
                                ignore_eos=True)
        torch.npu.synchronize()
        start = time.perf_counter()
        prompts = [
            {"prompt_token_ids": list(prompt_ids)} for _ in range(batch)]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.npu.synchronize()
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        return time.perf_counter() - start, tokens

    # Match auto-infer's cold phase: the first request after construction,
    # before any explicit warmup or measured steady-state request.
    cold_ttft_seconds, _ = run(1, 1)
    for _ in range(manifest["warmup_runs"]):
        run(1, 8)
        run(manifest["throughput_batch"], 8)
    ttft_samples, full_samples, throughput_samples = [], [], []
    throughput_elapsed_samples = []
    last_output = []
    for _ in range(manifest["measured_runs"]):
        elapsed, _ = run(1, 1)
        ttft_samples.append(elapsed)
        elapsed, outputs = run(1, manifest["output_tokens"])
        full_samples.append(elapsed)
        last_output = outputs[0]
        elapsed, outputs = run(manifest["throughput_batch"], manifest["output_tokens"])
        throughput_elapsed_samples.append(elapsed)
        throughput_samples.append(sum(map(len, outputs)) / elapsed)
    median_ttft = summarize(ttft_samples)["median"]
    median_full = summarize(full_samples)["median"]
    report = {
        "framework": framework,
        "manifest": manifest,
        "load_seconds": load_seconds,
        "cold_ttft_seconds": cold_ttft_seconds,
        "ttft_seconds": summarize(ttft_samples),
        "full_request_seconds": summarize(full_samples),
        "tpot_seconds": (median_full - median_ttft) / (manifest["output_tokens"] - 1),
        "throughput_tokens_per_second": summarize(throughput_samples),
        "peak_allocated_gib": torch.npu.max_memory_allocated() / 2 ** 30,
        "kv_capacity": {
            "usable_tokens": manifest["usable_kv_tokens"],
            "usable_blocks": runtime_blocks,
            "physical_blocks": runtime_blocks,
            "scratch_blocks": 0,
            "runtime_block_size": runtime_block_size,
        },
        "stability": {
            "throughput_cv": summarize(throughput_samples)[
                "coefficient_of_variation"],
            "request_elapsed_seconds": summarize(throughput_elapsed_samples),
        },
        "output_digest": token_digest(last_output),
        "output_length": len(last_output),
        "async_mode": {
            "enabled": runtime_async_scheduling(llm),
            "depth": "runtime-managed",
        },
        "path_counters": {"runtime_managed": True},
        "phase_samples": {
            "cold_ttft_seconds": [cold_ttft_seconds],
            "ttft_seconds": ttft_samples,
            "full_request_seconds": full_samples,
            "throughput_tokens_per_second": throughput_samples,
            "throughput_request_seconds": throughput_elapsed_samples,
        },
    }
    validate_comparison_result(report)
    emit_report(report)


if __name__ == "__main__":
    main()
