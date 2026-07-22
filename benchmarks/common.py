import hashlib
import json
import math
import os
import statistics
from pathlib import Path


REQUIRED_FIELDS = {
    "model", "prompt", "max_model_len", "output_tokens", "throughput_batch",
    "warmup_runs", "measured_runs", "dtype", "temperature", "seed",
    "usable_kv_tokens", "kv_block_size", "kv_cache_memory_bytes",
    "async_scheduling", "async_batches",
}

REQUIRED_RESULT_FIELDS = {
    "framework", "manifest", "load_seconds", "ttft_seconds",
    "full_request_seconds", "tpot_seconds", "throughput_tokens_per_second",
    "peak_allocated_gib", "output_digest", "output_length", "async_mode",
    "path_counters", "phase_samples", "kv_capacity", "stability",
    "cold_ttft_seconds",
}


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    missing = REQUIRED_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"comparison manifest missing fields: {sorted(missing)}")


def summarize(samples: list[float]) -> dict:
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median": statistics.median(samples),
        "mean": mean,
        "stdev": stdev,
        "count": len(samples),
        "coefficient_of_variation": stdev / abs(mean) if mean else 0.0,
        "samples": samples,
    }


def token_digest(token_ids: list[int]) -> str:
    payload = ",".join(str(token) for token in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_comparison_result(
        result: dict, *, allow_missing_cold: bool = False) -> None:
    required_fields = REQUIRED_RESULT_FIELDS
    if allow_missing_cold:
        required_fields = required_fields - {"cold_ttft_seconds"}
    missing = required_fields - result.keys()
    if missing:
        raise ValueError(f"comparison result missing fields: {sorted(missing)}")
    manifest = result["manifest"]
    validate_manifest(manifest)
    phases = result["phase_samples"]
    required_phases = {
        "ttft_seconds", "full_request_seconds", "throughput_tokens_per_second"}
    missing_phases = required_phases - phases.keys()
    if missing_phases:
        raise ValueError(f"phase_samples missing fields: {sorted(missing_phases)}")
    measured_runs = manifest["measured_runs"]
    wrong_counts = {
        name: len(phases[name]) for name in required_phases
        if len(phases[name]) != measured_runs}
    if wrong_counts:
        raise ValueError(
            f"phase sample counts do not match measured_runs={measured_runs}: "
            f"{wrong_counts}")
    if not allow_missing_cold or "cold_ttft_seconds" in result:
        cold_value = result["cold_ttft_seconds"]
        cold_samples = phases.get("cold_ttft_seconds")
        if cold_samples is None:
            raise ValueError(
                "phase_samples missing cold_ttft_seconds evidence")
        if (not isinstance(cold_value, (int, float))
                or isinstance(cold_value, bool)
                or not math.isfinite(cold_value) or cold_value <= 0
                or cold_samples != [cold_value]):
            raise ValueError(
                "cold_ttft_seconds must match one positive raw phase sample")
    for name in required_phases:
        if result[name] != summarize(phases[name]):
            raise ValueError(f"{name} summary does not match phase samples")
    if result["output_length"] != manifest["output_tokens"]:
        raise ValueError("output_length does not match manifest output_tokens")
    capacity = result["kv_capacity"]
    required_capacity = {
        "usable_tokens", "usable_blocks", "physical_blocks",
        "scratch_blocks", "runtime_block_size"}
    missing_capacity = required_capacity - capacity.keys()
    if missing_capacity:
        raise ValueError(
            f"kv_capacity missing fields: {sorted(missing_capacity)}")
    if capacity["usable_tokens"] != manifest["usable_kv_tokens"]:
        raise ValueError("result KV capacity does not match manifest")
    if (capacity["usable_blocks"] * capacity["runtime_block_size"]
            != capacity["usable_tokens"]):
        raise ValueError("usable KV block arithmetic is inconsistent")
    if (capacity["physical_blocks"]
            != capacity["usable_blocks"] + capacity["scratch_blocks"]):
        raise ValueError("physical KV block arithmetic is inconsistent")
    expected_cv = summarize(
        phases["throughput_tokens_per_second"])["coefficient_of_variation"]
    if result["stability"]["throughput_cv"] != expected_cv:
        raise ValueError("throughput stability does not match phase samples")
    if "throughput_request_seconds" in phases:
        expected_elapsed = summarize(phases["throughput_request_seconds"])
        if result["stability"]["request_elapsed_seconds"] != expected_elapsed:
            raise ValueError("request stability does not match phase samples")
    if (result["framework"] == "auto-infer"
            and result["async_mode"]["enabled"]
            != manifest["async_scheduling"]):
        raise ValueError("auto-infer async mode does not match manifest")
    if (result["framework"] == "auto-infer"
            and result["async_mode"]["depth"] != manifest["async_batches"]):
        raise ValueError("auto-infer async depth does not match manifest")


def validate_comparable_results(results: list[dict]) -> None:
    """Reject memory rankings that do not expose equal usable KV capacity."""
    capacities = {
        result["kv_capacity"]["usable_tokens"] for result in results}
    if len(capacities) != 1:
        raise ValueError(
            f"unequal usable KV capacity: {sorted(capacities)}")


def emit_report(report: dict) -> None:
    """Print the human log marker and optionally persist clean result JSON."""
    payload = json.dumps(report, sort_keys=True)
    result_path = os.environ.get("AUTO_INFER_BENCHMARK_RESULT")
    if result_path:
        Path(result_path).write_text(payload + "\n")
    print("FRAMEWORK_RESULT " + payload, flush=True)


def runtime_kv_block_size(llm, fallback: int) -> int:
    """Read vLLM's post-platform block size without depending on its types."""
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    cache = getattr(config, "cache_config", None)
    return int(getattr(cache, "block_size", fallback))


def runtime_async_scheduling(llm) -> bool | str:
    """Report runtime state when exposed; never infer it from framework name."""
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    enabled = getattr(scheduler, "async_scheduling", None)
    return bool(enabled) if enabled is not None else "unknown"
