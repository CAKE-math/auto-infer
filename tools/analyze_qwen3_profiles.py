"""Normalize Qwen3 Chrome traces and matched benchmark evidence."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_results import EXPECTED_FRAMEWORKS, load_comparable_results
from benchmarks.qwen3_profile_common import (
    load_chrome_trace_events, sha256_file, validate_chrome_trace)


PHASES = (
    "graph_replay",
    "attention_kv",
    "projection_mlp_norm",
    "lm_head_sampling",
    "runtime_scheduling",
    "communication_memory",
    "unclassified",
)

_PHASE_PATTERNS = {
    "graph_replay": ("graphreplay", "graph replay", "aclgraph", "graph_replay"),
    "attention_kv": (
        "attention", "paged_attention", "flash_attention", "kv_cache",
        "pa_kv", "scatter_pa", "reshape_and_cache", "rotary"),
    "projection_mlp_norm": (
        "grouped_matmul", "matmul", "linear", "gemm", "rms_norm",
        "rmsnorm", "layer_norm", "layernorm", "silu", "swiglu"),
    "lm_head_sampling": (
        "argmax", "multinomial", "softmax", "sampling", "sampler",
        "lm_head", "logits"),
    "runtime_scheduling": (
        "qwen3/profiled_request", "qwen3/prefill_and_decode",
        "auto-infer", "omni-npu", "vllm-ascend", "schedule", "enqueue",
        "dequeue", "launchtask", "launch task"),
    "communication_memory": (
        "alltoall", "all_to_all", "allreduce", "all_reduce", "hcom",
        "memcpy", "memset", "hosttodevice", "devicetohost", "copy_"),
}


def classify_event(name: str) -> str:
    normalized = name.casefold()
    for phase, patterns in _PHASE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return phase
    return "unclassified"


def _numeric_duration(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def summarize_trace(path: Path) -> dict:
    events = load_chrome_trace_events(path)
    phase_totals = {phase: {"count": 0, "duration_us": 0.0}
                    for phase in PHASES}
    names = defaultdict(lambda: {
        "count": 0, "duration_us": 0.0, "categories": set(),
        "pids": set(), "tids": set()})
    complete_count = 0
    total_duration = 0.0
    unclassified = set()
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        duration = _numeric_duration(event.get("dur"))
        if duration is None:
            continue
        name = str(event.get("name", "<unnamed>"))
        phase = classify_event(name)
        complete_count += 1
        total_duration += duration
        phase_totals[phase]["count"] += 1
        phase_totals[phase]["duration_us"] += duration
        entry = names[name]
        entry["count"] += 1
        entry["duration_us"] += duration
        if event.get("cat") is not None:
            entry["categories"].add(str(event["cat"]))
        if event.get("pid") is not None:
            entry["pids"].add(event["pid"])
        if event.get("tid") is not None:
            entry["tids"].add(event["tid"])
        if phase == "unclassified":
            unclassified.add(name)
    phases = {}
    for phase, values in phase_totals.items():
        duration = values["duration_us"]
        phases[phase] = {
            "count": values["count"],
            "duration_us": duration,
            "share": duration / total_duration if total_duration else 0.0,
        }
    top_events = []
    for name, values in names.items():
        top_events.append({
            "name": name,
            "count": values["count"],
            "duration_us": values["duration_us"],
            "categories": sorted(values["categories"]),
            "pids": sorted(values["pids"], key=str),
            "tids": sorted(values["tids"], key=str),
        })
    top_events.sort(key=lambda item: (-item["duration_us"], item["name"]))
    return {
        "event_count": len(events),
        "complete_event_count": complete_count,
        "total_event_duration_us": total_duration,
        "phases": phases,
        "top_events": top_events[:30],
        "unclassified_names": sorted(unclassified),
    }


def _without_framework(workload: dict) -> dict:
    normalized = {
        key: value for key, value in workload.items() if key != "framework"}
    output_tokens = normalized["output_tokens"]
    normalized.setdefault("capture_phases", {
        "prefill_passes": 1,
        "decode_passes": max(output_tokens - 1, 0),
        "continuous_decode": output_tokens > 1,
        "speculative_mtp": False,
    })
    return normalized


def _locate_trace(metadata_path: Path, output_dir: Path, framework: str,
                  metadata: dict) -> Path:
    candidates = [
        metadata_path.parent / metadata["trace"]["file"],
        output_dir / "raw" / f"{framework}.trace.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"trace not found for {framework}: {candidates}")


def build_evidence(metadata_paths: list[Path], benchmark_paths: list[Path],
                   output_dir: Path) -> tuple[dict, dict]:
    results = load_comparable_results(
        benchmark_paths, allow_missing_cold=True)
    result_by_framework = {result["framework"]: result for result in results}
    metadata_by_framework = {}
    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        framework = metadata.get("framework")
        if framework in metadata_by_framework:
            raise ValueError(f"duplicate profile metadata for {framework}")
        metadata_by_framework[framework] = (path, metadata)
    if set(metadata_by_framework) != EXPECTED_FRAMEWORKS:
        raise ValueError("profile metadata must cover all three frameworks")

    canonical_workload = None
    profiles = {}
    artifacts = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for framework in sorted(EXPECTED_FRAMEWORKS):
        metadata_path, metadata = metadata_by_framework[framework]
        workload = _without_framework(metadata["workload"])
        metadata = {**metadata, "workload": {
            **metadata["workload"],
            "capture_phases": workload["capture_phases"],
        }}
        if canonical_workload is None:
            canonical_workload = workload
        elif workload != canonical_workload:
            raise ValueError("profile workloads differ across frameworks")
        if metadata["output_length"] != metadata["workload"]["output_tokens"]:
            raise ValueError(f"profile output length mismatch for {framework}")
        trace_path = _locate_trace(
            metadata_path, output_dir, framework, metadata)
        trace_contract = validate_chrome_trace(trace_path)
        digest = sha256_file(trace_path)
        if digest != metadata["trace"]["sha256"]:
            raise ValueError(f"trace hash mismatch for {framework}")
        if trace_contract["event_count"] != metadata["trace"]["event_count"]:
            raise ValueError(f"trace event count mismatch for {framework}")
        profiles[framework] = summarize_trace(trace_path)
        artifacts[framework] = {
            "path": f"raw/{framework}.trace.json",
            "sha256": digest,
            **trace_contract,
            "metadata": metadata,
        }

    auto = result_by_framework["auto-infer"]
    relative = {}
    for competitor in ("omni-npu", "vllm-ascend"):
        other = result_by_framework[competitor]
        relative[competitor] = {
            "ttft_speedup": other["ttft_seconds"]["median"]
                            / auto["ttft_seconds"]["median"],
            "tpot_speedup": other["tpot_seconds"] / auto["tpot_seconds"],
            "throughput_speedup": (
                auto["throughput_tokens_per_second"]["median"]
                / other["throughput_tokens_per_second"]["median"]),
            "load_speedup": other["load_seconds"] / auto["load_seconds"],
            "allocation_ratio": (
                other["peak_allocated_gib"] / auto["peak_allocated_gib"]),
        }
    manifest = {
        "schema_version": 1,
        "workload": canonical_workload,
        "artifacts": artifacts,
        "benchmark_sources": {
            result["framework"]: str(path)
            for result, path in zip(results, benchmark_paths)
        },
        "profile_timing_semantics": (
            "Summed complete-event duration; concurrent streams overlap and "
            "values are not request wall-clock time."),
    }
    summary = {
        "schema_version": 1,
        "headline_benchmarks": result_by_framework,
        "relative_to_auto_infer": relative,
        "profiles": profiles,
        "accuracy": {
            "auto_infer_matches_vllm_ascend": (
                auto["output_digest"]
                == result_by_framework["vllm-ascend"]["output_digest"]),
            "omni_npu_digest_matches": (
                auto["output_digest"]
                == result_by_framework["omni-npu"]["output_digest"]),
            "all_output_lengths_match": len({
                result["output_length"] for result in results}) == 1,
        },
    }
    return manifest, summary


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", nargs=3, required=True, type=Path)
    parser.add_argument("--benchmarks", nargs=3, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    manifest, summary = build_evidence(
        args.metadata, args.benchmarks, args.output_dir)
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
