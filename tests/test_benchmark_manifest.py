import json
from pathlib import Path

import pytest

from benchmarks.common import (
    emit_report, load_manifest, runtime_async_scheduling,
    runtime_kv_block_size, summarize, token_digest,
    validate_comparable_results,
    validate_comparison_result)
from benchmarks.serving_common import (serving_result, summarize_latency,
                                       validate_serving_result)


def test_comparison_manifest_is_complete_and_deterministic():
    root = Path(__file__).parents[1]
    manifest = load_manifest(root / "benchmarks/comparison_manifest.json")
    assert manifest["temperature"] == 0.0
    assert manifest["warmup_runs"] >= 1
    assert manifest["measured_runs"] >= 3
    assert manifest["throughput_batch"] > 1
    assert summarize([1.0, 2.0, 3.0])["median"] == 2.0
    assert summarize([1.0, 2.0, 3.0])["count"] == 3
    assert summarize([1.0, 2.0, 3.0])["coefficient_of_variation"] == 0.5
    assert summarize([0.0, 0.0])["coefficient_of_variation"] == 0.0
    assert manifest["usable_kv_tokens"] % manifest["kv_block_size"] == 0
    assert manifest["async_scheduling"] is False
    assert manifest["async_batches"] == 2
    assert token_digest([1, 2, 3]) == token_digest([1, 2, 3])


def test_comparison_result_requires_execution_evidence():
    manifest = {
        "model": "qwen", "prompt": "hello", "max_model_len": 2048,
        "output_tokens": 128, "throughput_batch": 16, "warmup_runs": 1,
        "measured_runs": 1, "dtype": "bfloat16", "temperature": 0.0,
        "seed": 0, "usable_kv_tokens": 14464, "kv_block_size": 16,
        "kv_cache_memory_bytes": 1658847232,
        "async_scheduling": False, "async_batches": 2,
    }
    result = {
        "framework": "auto-infer", "manifest": manifest, "load_seconds": 1.0,
        "ttft_seconds": summarize([0.1]),
        "full_request_seconds": summarize([1.0]),
        "tpot_seconds": 0.01,
        "throughput_tokens_per_second": summarize([100.0]),
        "peak_allocated_gib": 1.0, "output_digest": "abc",
        "output_length": 128,
    }
    with pytest.raises(ValueError, match="async_mode.*path_counters.*phase_samples"):
        validate_comparison_result(result)

    result.update({
        "async_mode": {"enabled": False, "depth": 2},
        "path_counters": {"graph_steps": 127},
        "kv_capacity": {
            "usable_tokens": 14464, "usable_blocks": 904,
            "physical_blocks": 936, "scratch_blocks": 32,
            "runtime_block_size": 16},
        "stability": {
            "throughput_cv": 0.0,
            "request_elapsed_seconds": summarize([1.0, 1.1])},
        "cold_ttft_seconds": 0.25,
        "phase_samples": {
            "cold_ttft_seconds": [0.25],
            "ttft_seconds": [0.1], "full_request_seconds": [1.0],
            "throughput_tokens_per_second": [100.0]},
    })
    validate_comparison_result(result)

    result["phase_samples"]["cold_ttft_seconds"] = [0.24]
    with pytest.raises(ValueError, match="cold_ttft_seconds"):
        validate_comparison_result(result)
    result["phase_samples"]["cold_ttft_seconds"] = [0.25]

    result["cold_ttft_seconds"] = float("inf")
    result["phase_samples"]["cold_ttft_seconds"] = [float("inf")]
    with pytest.raises(ValueError, match="cold_ttft_seconds"):
        validate_comparison_result(result)
    result["cold_ttft_seconds"] = 0.25
    result["phase_samples"]["cold_ttft_seconds"] = [0.25]

    result["output_length"] = 127
    with pytest.raises(ValueError, match="output_length"):
        validate_comparison_result(result)

    result["output_length"] = 128
    result["kv_capacity"]["physical_blocks"] = 935
    with pytest.raises(ValueError, match="physical KV block arithmetic"):
        validate_comparison_result(result)


def test_comparison_rejects_unequal_usable_kv_capacity():
    def result(tokens):
        return {"framework": str(tokens),
                "kv_capacity": {"usable_tokens": tokens}}

    validate_comparable_results([result(14464), result(14464)])
    with pytest.raises(ValueError, match="unequal usable KV capacity"):
        validate_comparable_results([result(14464), result(65536)])


def test_runtime_kv_block_size_reads_engine_config_with_safe_fallback():
    from types import SimpleNamespace
    llm = SimpleNamespace(llm_engine=SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128))))

    assert runtime_kv_block_size(llm, fallback=16) == 128
    assert runtime_kv_block_size(SimpleNamespace(), fallback=16) == 16


def test_runtime_async_scheduling_reads_config_without_guessing():
    from types import SimpleNamespace
    llm = SimpleNamespace(llm_engine=SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=True))))

    assert runtime_async_scheduling(llm) is True
    assert runtime_async_scheduling(SimpleNamespace()) == "unknown"


def test_emit_report_persists_machine_readable_result(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("AUTO_INFER_BENCHMARK_RESULT", str(result_path))

    emit_report({"framework": "auto-infer", "value": 3})

    assert json.loads(result_path.read_text()) == {
        "framework": "auto-infer", "value": 3}


def test_serving_result_requires_workload_identity_and_raw_samples():
    result = serving_result(
        framework="auto-infer",
        model="moonlight",
        git_commit="abc123",
        workload={
            "prompt_tokens": 128,
            "output_tokens": 32,
            "arrival_rate": 4.0,
            "concurrency": 16,
            "warmup_requests": 2,
            "measured_requests": 4,
        },
        ttft_samples=[0.1, 0.2, 0.3, 0.4],
        itl_samples=[0.01, 0.02],
        e2e_samples=[1.0, 1.1, 1.2, 1.3],
        elapsed_seconds=2.0,
        completed=4,
        rejected=0,
        failed=0,
        client_cpu_seconds=0.5,
        client_peak_rss_bytes=1024,
        server_cpu_seconds=0.25,
        server_peak_rss_bytes=2048,
    )

    validate_serving_result(result)
    assert result["latency"]["ttft"]["p50"] == 0.25
    assert result["request_throughput_per_second"] == 2.0
    assert result["output_throughput_tokens_per_second"] == 64.0
    assert result["samples"]["ttft_seconds"] == [0.1, 0.2, 0.3, 0.4]
    assert result["server_cpu_seconds"] == 0.25
    assert result["server_peak_rss_bytes"] == 2048

    del result["git_commit"]
    with pytest.raises(ValueError, match="git_commit"):
        validate_serving_result(result)


def test_latency_summary_has_p50_p95_and_p99():
    summary = summarize_latency([float(value) for value in range(1, 101)])

    assert summary == {"p50": 50.5, "p95": 95.05, "p99": 99.01}
