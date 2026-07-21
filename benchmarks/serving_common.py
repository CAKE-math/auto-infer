"""Shared result schema for reproducible online Serving benchmarks."""

import math
import uuid


_WORKLOAD_FIELDS = {
    "prompt_tokens",
    "output_tokens",
    "arrival_rate",
    "concurrency",
    "warmup_requests",
    "measured_requests",
}

_RESULT_FIELDS = {
    "run_id",
    "framework",
    "model",
    "git_commit",
    "workload",
    "latency",
    "request_throughput_per_second",
    "output_throughput_tokens_per_second",
    "client_cpu_seconds",
    "client_peak_rss_bytes",
    "server_cpu_seconds",
    "server_peak_rss_bytes",
    "completed",
    "rejected",
    "failed",
    "rejection_rate",
    "error_rate",
    "samples",
}


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("latency samples must not be empty")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latency(samples: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def serving_result(*, framework: str, model: str, git_commit: str,
                   workload: dict, ttft_samples: list[float],
                   itl_samples: list[float], e2e_samples: list[float],
                   elapsed_seconds: float, completed: int, rejected: int,
                   failed: int, client_cpu_seconds: float,
                   client_peak_rss_bytes: int,
                   server_cpu_seconds: float,
                   server_peak_rss_bytes: int) -> dict:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be > 0")
    total = completed + rejected + failed
    output_tokens = int(workload["output_tokens"])
    result = {
        "run_id": uuid.uuid4().hex,
        "framework": framework,
        "model": model,
        "git_commit": git_commit,
        "workload": dict(workload),
        "latency": {
            "ttft": summarize_latency(ttft_samples),
            "itl": summarize_latency(itl_samples),
            "e2e": summarize_latency(e2e_samples),
        },
        "request_throughput_per_second": completed / elapsed_seconds,
        "output_throughput_tokens_per_second": (
            completed * output_tokens / elapsed_seconds
        ),
        "client_cpu_seconds": client_cpu_seconds,
        "client_peak_rss_bytes": int(client_peak_rss_bytes),
        "server_cpu_seconds": server_cpu_seconds,
        "server_peak_rss_bytes": server_peak_rss_bytes,
        "completed": completed,
        "rejected": rejected,
        "failed": failed,
        "rejection_rate": rejected / total if total else 0.0,
        "error_rate": failed / total if total else 0.0,
        "samples": {
            "ttft_seconds": list(ttft_samples),
            "itl_seconds": list(itl_samples),
            "e2e_seconds": list(e2e_samples),
        },
    }
    validate_serving_result(result)
    return result


def validate_serving_result(result: dict) -> None:
    missing = _RESULT_FIELDS - result.keys()
    if missing:
        raise ValueError(f"serving result missing fields: {sorted(missing)}")
    missing_workload = _WORKLOAD_FIELDS - result["workload"].keys()
    if missing_workload:
        raise ValueError(
            f"serving workload missing fields: {sorted(missing_workload)}"
        )
    for identity in ("run_id", "framework", "model", "git_commit"):
        if not result[identity]:
            raise ValueError(f"{identity} must not be empty")
    samples = result["samples"]
    for name in ("ttft_seconds", "itl_seconds", "e2e_seconds"):
        values = samples.get(name)
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"{name} must contain finite non-negative samples")
    submitted = int(result["workload"]["measured_requests"])
    accounted = result["completed"] + result["rejected"] + result["failed"]
    if submitted != accounted:
        raise ValueError(
            f"request accounting mismatch: submitted={submitted}, accounted={accounted}"
        )
    if (not math.isfinite(result["server_cpu_seconds"])
            or result["server_cpu_seconds"] < 0):
        raise ValueError("server_cpu_seconds must be finite and non-negative")
    if result["server_peak_rss_bytes"] <= 0:
        raise ValueError("server_peak_rss_bytes must be > 0")
