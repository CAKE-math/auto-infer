"""Shared artifact contract for the bounded Qwen3 profiler capture."""

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path


REQUIRED_METADATA_FIELDS = {
    "framework",
    "trace",
    "workload",
    "environment",
    "output_digest",
    "output_length",
}

_PREFILL_MARKER = "qwen3/phase/prefill"
_DECODE_MARKER = re.compile(r"qwen3/phase/decode/(\d{3})$")
_PHASE_CATEGORY = "qwen3.phase"


class StepPhaseRecorder:
    """Annotate one engine step as prefill and every later step as decode."""

    def __init__(self, record_function):
        self._record_function = record_function
        self.step_count = 0

    @contextmanager
    def instrument(self, engine, output_tokens: int):
        original = engine.step
        had_instance_step = "step" in vars(engine)

        def annotated_step(*args, **kwargs):
            index = self.step_count
            if index == 0:
                name = _PREFILL_MARKER
            elif index < output_tokens:
                name = f"qwen3/phase/decode/{index:03d}"
            else:
                name = f"qwen3/runtime/drain/{index - output_tokens + 1:03d}"
            with self._record_function(name):
                result = original(*args, **kwargs)
            self.step_count += 1
            return result

        engine.step = annotated_step
        try:
            yield self
        finally:
            if had_instance_step:
                engine.step = original
            else:
                del engine.step

    def validate(self, output_tokens: int) -> dict:
        expected = max(int(output_tokens), 0)
        if self.step_count < expected:
            raise ValueError(
                "profiled engine step count is below output tokens: "
                f"steps={self.step_count}, output_tokens={expected}")
        return {
            "prefill_passes": 1 if expected else 0,
            "decode_passes": max(expected - 1, 0),
            "runtime_drains": self.step_count - expected,
        }


def load_chrome_trace_events(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    return events


def _phase_source_events(events: list[dict]) -> list[tuple[str, int, dict]]:
    markers = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        if name == _PREFILL_MARKER:
            markers.append(("prefill", 0, event))
            continue
        match = _DECODE_MARKER.fullmatch(name)
        if match:
            markers.append(("decode", int(match.group(1)), event))
    return sorted(markers, key=lambda item: item[1])


def add_visible_phase_lane(path: Path, output_tokens: int) -> dict:
    """Add a dedicated Chrome lane copied from profiler phase ranges."""
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    markers = _phase_source_events(events)
    expected = [("prefill", 0)] + [
        ("decode", step) for step in range(1, output_tokens)]
    actual = [(phase, step) for phase, step, _ in markers]
    if actual != expected:
        raise ValueError(
            f"phase marker mismatch: expected={expected}, actual={actual}")

    native_pids = [event.get("pid") for event in events
                   if isinstance(event, dict)
                   and isinstance(event.get("pid"), int)]
    lane_pid = (max(native_pids) + 1) if native_pids else 1
    lane_tid = 1
    lane_events = [
        {"name": "process_name", "ph": "M", "pid": lane_pid, "tid": 0,
         "args": {"name": "QWEN3 PHASES"}},
        {"name": "process_sort_index", "ph": "M", "pid": lane_pid,
         "tid": 0, "args": {"sort_index": -1000}},
        {"name": "thread_name", "ph": "M", "pid": lane_pid,
         "tid": lane_tid, "args": {"name": "PREFILL + DECODE STEPS"}},
    ]
    steps = []
    for phase, step, source in markers:
        label = "PREFILL" if phase == "prefill" else f"DECODE {step:03d}"
        duration = float(source["dur"])
        timestamp = float(source["ts"])
        lane_events.append({
            "name": label,
            "cat": _PHASE_CATEGORY,
            "ph": "X",
            "ts": timestamp,
            "dur": duration,
            "pid": lane_pid,
            "tid": lane_tid,
            "args": {
                "phase": phase,
                "step": step,
                "source_name": source["name"],
                "source_pid": source.get("pid"),
                "source_tid": source.get("tid"),
            },
        })
        steps.append({
            "phase": phase,
            "step": step,
            "label": label,
            "timestamp_us": timestamp,
            "duration_us": duration,
        })
    events.extend(lane_events)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"schema_version": 1, "lane": "QWEN3 PHASES", "steps": steps}


def extract_phase_index(path: Path) -> dict:
    steps = []
    for event in load_chrome_trace_events(path):
        if (not isinstance(event, dict) or event.get("ph") != "X"
                or event.get("cat") != _PHASE_CATEGORY):
            continue
        args = event.get("args", {})
        steps.append({
            "phase": args["phase"],
            "step": int(args["step"]),
            "label": event["name"],
            "timestamp_us": float(event["ts"]),
            "duration_us": float(event["dur"]),
        })
    steps.sort(key=lambda item: item["step"])
    if not steps:
        raise ValueError("Chrome trace lacks the QWEN3 PHASES lane")
    return {"schema_version": 1, "lane": "QWEN3 PHASES", "steps": steps}


def validate_chrome_trace(path: Path) -> dict:
    events = load_chrome_trace_events(path)
    return {"event_count": len(events), "size_bytes": path.stat().st_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_metadata(path: Path, metadata: dict) -> None:
    missing = REQUIRED_METADATA_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"profile metadata missing fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
