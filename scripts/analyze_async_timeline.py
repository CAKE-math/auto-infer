#!/usr/bin/env python3
"""Validate zero-host-bubble ordering in a Chrome profiler trace."""
import argparse
import json
import statistics
from pathlib import Path


def _events(payload):
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def analyze(payload, sample_pattern="argmax"):
    submits = {}
    samples = []
    clones = []
    device_events = []
    for event in _events(payload):
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        timestamp = float(event.get("ts", 0))
        duration = float(event.get("dur", 0))
        if name.startswith("auto_infer.async/") and name.endswith("/submit"):
            sequence = int(name.split("/")[1])
            submits[sequence] = timestamp + duration
        lowered = name.lower()
        if sample_pattern.lower() in lowered:
            samples.append((timestamp + duration, name))
        if "clone" in lowered:
            clones.append(name)
        category = str(event.get("cat", "")).lower()
        if any(token in category for token in ("npu", "kernel", "device")):
            device_events.append((timestamp, timestamp + duration, name))

    samples.sort()
    ordering = []
    for sequence, (sample_end, _) in enumerate(samples[:-1]):
        next_submit = submits.get(sequence + 1)
        if next_submit is not None:
            ordering.append({
                "step": sequence,
                "submit_next_end_us": next_submit,
                "sample_end_us": sample_end,
                "pass": next_submit < sample_end,
            })

    device_events.sort()
    gaps = [
        max(0.0, right[0] - left[1])
        for left, right in zip(device_events, device_events[1:])
        if right[0] >= left[0]
    ]
    return {
        "markers_present": bool(submits and samples),
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
    args = parser.parse_args()
    verdict = analyze(
        json.loads(args.trace.read_text()), args.sample_pattern)
    print(json.dumps(verdict, indent=2))
    raise SystemExit(
        0 if verdict["markers_present"] and verdict["ordering_pass"]
        and verdict["clone_free"] else 1)


if __name__ == "__main__":
    main()
