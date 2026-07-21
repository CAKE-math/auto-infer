import json

import pytest

from benchmarks.compare_results import load_comparable_results
from benchmarks.common import summarize


def _manifest():
    return {
        "model": "qwen", "prompt": "hello", "max_model_len": 2048,
        "output_tokens": 8, "throughput_batch": 16, "warmup_runs": 1,
        "measured_runs": 1, "dtype": "bfloat16", "temperature": 0.0,
        "seed": 0, "usable_kv_tokens": 14464, "kv_block_size": 16,
        "kv_cache_memory_bytes": 1658847232,
        "async_scheduling": False, "async_batches": 2,
    }


def _result(framework, tokens=14464, manifest=None):
    return {
        "framework": framework,
        "manifest": manifest or _manifest(),
        "load_seconds": 1.0,
        "cold_ttft_seconds": 0.2,
        "ttft_seconds": summarize([0.1]),
        "full_request_seconds": summarize([1.0]),
        "tpot_seconds": 0.1,
        "throughput_tokens_per_second": summarize([100.0]),
        "peak_allocated_gib": 2.0,
        "output_digest": "abc",
        "output_length": 8,
        "async_mode": {"enabled": False, "depth": 2},
        "path_counters": {},
        "phase_samples": {
            "cold_ttft_seconds": [0.2],
            "ttft_seconds": [0.1],
            "full_request_seconds": [1.0],
            "throughput_tokens_per_second": [100.0],
        },
        "kv_capacity": {
            "usable_tokens": tokens,
            "usable_blocks": tokens // 16,
            "physical_blocks": tokens // 16,
            "scratch_blocks": 0,
            "runtime_block_size": 16,
        },
        "stability": {"throughput_cv": 0.0},
    }


def test_aggregate_comparison_enforces_capacity_and_frameworks(tmp_path):
    paths = []
    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        path = tmp_path / f"{framework}.json"
        path.write_text(json.dumps(_result(framework)))
        paths.append(path)

    results = load_comparable_results(paths)

    assert {result["framework"] for result in results} == {
        "auto-infer", "omni-npu", "vllm-ascend"}


def test_aggregate_comparison_rejects_capacity_not_backed_by_manifest(tmp_path):
    paths = []
    for index, (framework, tokens) in enumerate(zip(
            ("auto-infer", "omni-npu", "vllm-ascend"),
            (14464, 14464, 16384))):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(_result(framework, tokens)))
        paths.append(path)

    with pytest.raises(ValueError, match="KV capacity"):
        load_comparable_results(paths)


def test_aggregate_comparison_rejects_mismatched_manifests(tmp_path):
    paths = []
    for index, framework in enumerate(
            ("auto-infer", "omni-npu", "vllm-ascend")):
        path = tmp_path / f"result-{index}.json"
        manifest = _manifest()
        manifest["prompt"] = f"prompt-{index}"
        path.write_text(json.dumps(_result(framework, manifest=manifest)))
        paths.append(path)

    with pytest.raises(ValueError, match="benchmark manifests differ"):
        load_comparable_results(paths)
