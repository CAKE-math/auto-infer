import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                               ModelConfig, SchedulerConfig)
from auto_infer.entrypoints.llm import LLM
from benchmarks.common import (
    emit_report, load_manifest, summarize, token_digest,
    validate_comparison_result)


def main():
    manifest = load_manifest(sys.argv[1])
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"], trust_remote_code=True)
    prompt_ids = tokenizer(manifest["prompt"]).input_ids
    block_size = manifest["kv_block_size"]
    if manifest["usable_kv_tokens"] % block_size:
        raise ValueError("usable_kv_tokens must be divisible by kv_block_size")
    usable_blocks = manifest["usable_kv_tokens"] // block_size
    config = EngineConfig(
        model=ModelConfig(manifest["model"], max_model_len=manifest["max_model_len"],
                          dtype=manifest["dtype"]),
        cache=CacheConfig(block_size=block_size, num_blocks=usable_blocks),
        scheduler=SchedulerConfig(max_num_seqs=256, max_num_batched_tokens=8192),
        execution=ExecutionConfig(mode="graph", device_index=0, max_gear=32),
        async_scheduling=manifest["async_scheduling"],
        async_batches=manifest["async_batches"])
    torch.npu.reset_peak_memory_stats()
    llm = LLM(config)
    load_seconds = time.perf_counter() - load_start

    def run(batch, output_tokens):
        torch.npu.synchronize()
        start = time.perf_counter()
        outputs = llm.generate([list(prompt_ids) for _ in range(batch)],
                               max_tokens=output_tokens)
        torch.npu.synchronize()
        return time.perf_counter() - start, outputs

    # The first post-construction request is the reproducible cold-TTFT phase.
    # It intentionally runs before any benchmark warmup or measured request.
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
    runner = llm.engine.executor.runner
    decode_copied_block_rows = sum(
        gear.stager.copied_block_rows for gear in runner.gears.values())
    decode_copied_block_elements = sum(
        gear.stager.copied_block_elements for gear in runner.gears.values())
    prefill_copied_block_rows = sum(
        gear.stager.copied_block_rows for gear in runner.prefill_gears.values())
    prefill_copied_block_elements = sum(
        gear.stager.copied_block_elements
        for gear in runner.prefill_gears.values())
    path_counters = dict(runner.stats)
    path_counters.update(
        captured_gears=sorted(runner.gears),
        captured_prefill_gears=sorted(runner.prefill_gears),
        decode_copied_block_rows=decode_copied_block_rows,
        decode_copied_block_elements=decode_copied_block_elements,
        prefill_copied_block_rows=prefill_copied_block_rows,
        prefill_copied_block_elements=prefill_copied_block_elements,
        copied_block_rows=(
            decode_copied_block_rows + prefill_copied_block_rows),
        copied_block_elements=(
            decode_copied_block_elements + prefill_copied_block_elements),
    )
    report = {
        "framework": "auto-infer",
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
            "usable_blocks": usable_blocks,
            "physical_blocks": usable_blocks + runner.scratch_blocks,
            "scratch_blocks": runner.scratch_blocks,
            "runtime_block_size": block_size,
        },
        "stability": {
            "throughput_cv": summarize(throughput_samples)[
                "coefficient_of_variation"],
            "request_elapsed_seconds": summarize(throughput_elapsed_samples),
        },
        "output_digest": token_digest(last_output),
        "output_length": len(last_output),
        "async_mode": {
            "enabled": config.async_scheduling,
            "depth": config.async_batches,
        },
        "path_counters": path_counters,
        "phase_samples": {
            "cold_ttft_seconds": [cold_ttft_seconds],
            "ttft_seconds": ttft_samples,
            "full_request_seconds": full_samples,
            "throughput_tokens_per_second": throughput_samples,
            "throughput_request_seconds": throughput_elapsed_samples,
        },
    }
    validate_comparison_result(report)
    llm.close()
    emit_report(report)


if __name__ == "__main__":
    main()
