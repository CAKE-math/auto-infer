#!/usr/bin/env python3
"""Validate zero-host-bubble ordering in a Chrome profiler trace."""
import argparse
import json
import statistics
from pathlib import Path


def _events(payload):
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def analyze(
        payload, sample_pattern="argmax",
        graph_start_pattern="aclnnindex_indexaicore_index"):
    submits = {}
    task_updates = {}
    sample_candidates = []
    clones = []
    device_events = []
    graph_starts = []
    for event in _events(payload):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        timestamp = float(event.get("ts", 0))
        duration = float(event.get("dur", 0))
        if name.startswith("auto_infer.async/") and name.endswith("/submit"):
            sequence = int(name.split("/")[1])
            submits[sequence] = timestamp + duration
        if (name.startswith("auto_infer.async/")
                and name.endswith("/task_update")):
            sequence = int(name.split("/")[1])
            task_updates[sequence] = timestamp + duration
        lowered = name.lower()
        if sample_pattern.lower() in lowered:
            sample_candidates.append((timestamp + duration, name))
        if "clone" in lowered:
            clones.append((timestamp, name))
        if graph_start_pattern.lower() in lowered:
            graph_starts.append(timestamp)
        category = str(event.get("cat", "")).lower()
        if any(token in category for token in ("npu", "kernel", "device")):
            device_events.append((timestamp, timestamp + duration, name))

    # One graph sample emits several CPU/enqueue/cast events. The actual
    # ArgMaxV2 AiCore kernel is the unique per-step device boundary.
    samples = [
        row for row in sample_candidates
        if "argmaxv2aicore_argmaxv2" in row[1].lower()]
    if not samples:
        samples = sample_candidates
    samples.sort()
    first_sample_end = samples[0][0] if samples else float("-inf")
    clones = [
        name for timestamp, name in clones
        if timestamp > first_sample_end]
    ordering = []
    first_sequence = min(submits, default=0)
    for sequence, (sample_end, _) in enumerate(samples[:-1]):
        if sequence == 0:  # prefill is an explicit async barrier
            continue
        next_submit = submits.get(first_sequence + sequence + 1)
        next_update = task_updates.get(first_sequence + sequence + 1)
        if next_submit is not None:
            next_sample_end = samples[sequence + 1][0]
            ordering.append({
                "step": sequence,
                "submit_next_end_us": next_submit,
                "sample_end_us": sample_end,
                "task_update_next_end_us": next_update,
                "next_sample_end_us": next_sample_end,
                "task_update_hidden": (
                    next_update is not None
                    and next_update < next_sample_end),
                "pass": (
                    next_submit < sample_end
                    and next_update is not None
                    and next_update < next_sample_end),
            })

    graph_starts.sort()
    gaps = []
    for sample_end, _ in samples[1:-1]:
        next_start = next(
            (start for start in graph_starts if start >= sample_end), None)
        if next_start is not None:
            gaps.append(next_start - sample_end)
    return {
        "markers_present": bool(submits and task_updates and samples),
        "ordering": ordering,
        "ordering_pass": bool(ordering) and all(row["pass"] for row in ordering),
        "clone_free": not clones,
        "clone_events": clones,
        "device_gap_p50_us": statistics.median(gaps) if gaps else None,
        "device_gap_p95_us": (
            sorted(gaps)[min(len(gaps) - 1, int(len(gaps) * 0.95))]
            if gaps else None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--sample-pattern", default="argmax")
    parser.add_argument(
        "--graph-start-pattern", default="aclnnindex_indexaicore_index")
    args = parser.parse_args()
    verdict = analyze(
        json.loads(args.trace.read_text()), args.sample_pattern,
        args.graph_start_pattern)
    print(json.dumps(verdict, indent=2))
    raise SystemExit(
        0 if verdict["markers_present"] and verdict["ordering_pass"]
        and verdict["clone_free"] else 1)


if __name__ == "__main__":
    main()
