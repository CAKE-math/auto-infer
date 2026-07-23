"""Shared artifact contract for the bounded Qwen3 profiler capture."""

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path


def validate_auto_prefill_path(call_stack_index: dict) -> None:
    prefill_layers = [
        event["layer"]
        for event in call_stack_index.get("phases", {}).get("prefill", [])
    ]
    if prefill_layers.count("prefill-graph") != 1 or "eager" in prefill_layers:
        raise ValueError(
            "auto-infer prefill graph path mismatch: "
            f"observed layers={prefill_layers}")


def validate_auto_prefill_counters(counters: dict) -> None:
    expected = {
        "prefill_graph_steps": 1,
        "eager_steps": 0,
        "prefill_graph_online_captures": 0,
    }
    if any(counters.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "auto-infer prefill graph counters mismatch: "
            f"expected={expected}, observed={counters}")


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
_RUNTIME_MARKER = re.compile(r"qwen3/runtime/drain/(\d{3})$")
_CALL_MARKER = re.compile(r"qwen3/call/([^/]+)/(.+)$")
_PHASE_CATEGORY = "qwen3.phase"
_CALL_STACK_CATEGORY = "qwen3.callstack"


@dataclass(frozen=True)
class CallTarget:
    """One live object boundary to expose in the Chrome trace."""

    layer: str
    target: object
    method: str

    @property
    def symbol(self) -> str:
        owner = type(self.target)
        return f"{owner.__module__}.{owner.__qualname__}.{self.method}"


class RuntimeCallStackRecorder:
    """Wrap real runtime methods with profiler ranges, then restore them."""

    def __init__(self, record_function):
        self._record_function = record_function
        self.counts = {}

    @contextmanager
    def instrument(self, targets):
        installed = []
        seen = set()
        try:
            for target in targets:
                if "/" in target.layer:
                    raise ValueError("call-stack layer must not contain '/'")
                key = (id(target.target), target.method)
                if key in seen:
                    raise ValueError(f"duplicate call target: {target.symbol}")
                seen.add(key)
                original = getattr(target.target, target.method)
                instance_dict = getattr(target.target, "__dict__", {})
                had_instance_method = target.method in instance_dict
                name = f"qwen3/call/{target.layer}/{target.symbol}"

                @wraps(original)
                def traced(*args, __original=original, __name=name,
                           __layer=target.layer, **kwargs):
                    self.counts[__layer] = self.counts.get(__layer, 0) + 1
                    with self._record_function(__name):
                        return __original(*args, **kwargs)

                setattr(target.target, target.method, traced)
                installed.append((target, original, had_instance_method))
            yield self
        finally:
            for target, original, had_instance_method in reversed(installed):
                if had_instance_method:
                    setattr(target.target, target.method, original)
                else:
                    delattr(target.target, target.method)


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


def _runtime_phase_source_events(events: list[dict]):
    markers = list(_phase_source_events(events))
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        match = _RUNTIME_MARKER.fullmatch(str(event.get("name", "")))
        if match:
            markers.append(("runtime", int(match.group(1)), event))
    return sorted(markers, key=lambda item: float(item[2]["ts"]))


def _call_source_events(events: list[dict]):
    calls = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        match = _CALL_MARKER.fullmatch(str(event.get("name", "")))
        if match:
            calls.append((match.group(1), match.group(2), event))
    return calls


def _nested_call_events(calls, marker):
    start = float(marker["ts"])
    end = start + float(marker["dur"])
    pid, tid = marker.get("pid"), marker.get("tid")
    nested = []
    for layer, symbol, event in calls:
        timestamp = float(event["ts"])
        event_end = timestamp + float(event["dur"])
        if (event.get("pid"), event.get("tid")) != (pid, tid):
            continue
        if timestamp >= start and event_end <= end:
            nested.append((layer, symbol, event))
    nested.sort(key=lambda item: (
        float(item[2]["ts"]), -float(item[2]["dur"]), item[0]))

    stack = []
    result = []
    for layer, symbol, event in nested:
        timestamp = float(event["ts"])
        event_end = timestamp + float(event["dur"])
        while stack and (timestamp >= stack[-1] or event_end > stack[-1]):
            stack.pop()
        result.append({
            "layer": layer,
            "symbol": symbol,
            "timestamp_us": timestamp,
            "duration_us": float(event["dur"]),
            "depth": len(stack),
            "source_pid": event.get("pid"),
            "source_tid": event.get("tid"),
        })
        stack.append(event_end)
    return result


def add_visible_call_stack_lane(path: Path) -> dict:
    """Copy measured runtime boundaries into a dedicated nested trace lane."""
    payload = json.loads(path.read_text())
    events = payload if isinstance(payload, list) else payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    markers = _runtime_phase_source_events(events)
    calls = _call_source_events(events)
    if not calls:
        raise ValueError("Chrome trace lacks qwen3/call runtime ranges")

    native_pids = [event.get("pid") for event in events
                   if isinstance(event, dict)
                   and isinstance(event.get("pid"), int)]
    lane_pid = (max(native_pids) + 1) if native_pids else 1
    lane_tid = 1
    lane_events = [
        {"name": "process_name", "ph": "M", "pid": lane_pid, "tid": 0,
         "args": {"name": "QWEN3 CALL STACK"}},
        {"name": "process_sort_index", "ph": "M", "pid": lane_pid,
         "tid": 0, "args": {"sort_index": -999}},
        {"name": "thread_name", "ph": "M", "pid": lane_pid,
         "tid": lane_tid, "args": {"name": "TRACE-DERIVED RUNTIME CALLS"}},
    ]
    phases = {"prefill": [], "decode": [], "runtime": []}
    for phase, step, marker in markers:
        nested = _nested_call_events(calls, marker)
        if not nested:
            raise ValueError(f"phase {phase}:{step} lacks runtime call ranges")
        for item in nested:
            lane_events.append({
                "name": f'{item["layer"]} · {item["symbol"]}',
                "cat": _CALL_STACK_CATEGORY,
                "ph": "X",
                "ts": item["timestamp_us"],
                "dur": item["duration_us"],
                "pid": lane_pid,
                "tid": lane_tid,
                "args": {
                    "phase": phase,
                    "step": step,
                    "layer": item["layer"],
                    "symbol": item["symbol"],
                    "depth": item["depth"],
                    "source_pid": item["source_pid"],
                    "source_tid": item["source_tid"],
                },
            })
        if phase == "prefill":
            phases["prefill"] = nested
        else:
            phases[phase].append({
                "step": step,
                "duration_us": float(marker["dur"]),
                "events": nested,
            })
    events.extend(lane_events)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return {
        "schema_version": 1,
        "lane": "QWEN3 CALL STACK",
        "phases": phases,
    }


def extract_call_stack_index(path: Path) -> dict:
    events = load_chrome_trace_events(path)
    marker_durations = {
        (phase, step): float(marker["dur"])
        for phase, step, marker in _runtime_phase_source_events(events)
    }
    grouped = {"prefill": {}, "decode": {}, "runtime": {}}
    for event in events:
        if (not isinstance(event, dict) or event.get("ph") != "X"
                or event.get("cat") != _CALL_STACK_CATEGORY):
            continue
        args = event.get("args", {})
        phase, step = args["phase"], int(args["step"])
        grouped[phase].setdefault(step, []).append({
            "layer": args["layer"],
            "symbol": args["symbol"],
            "timestamp_us": float(event["ts"]),
            "duration_us": float(event["dur"]),
            "depth": int(args["depth"]),
            "source_pid": args.get("source_pid"),
            "source_tid": args.get("source_tid"),
        })
    if not any(grouped[phase] for phase in grouped):
        raise ValueError("Chrome trace lacks the QWEN3 CALL STACK lane")
    for by_step in grouped.values():
        for calls in by_step.values():
            calls.sort(key=lambda item: (
                item["timestamp_us"], -item["duration_us"], item["layer"]))
    phases = {
        "prefill": grouped["prefill"].get(0, []),
        "decode": [
            {"step": step,
             "duration_us": marker_durations[("decode", step)],
             "events": events_for_step}
            for step, events_for_step in sorted(grouped["decode"].items())
        ],
        "runtime": [
            {"step": step,
             "duration_us": marker_durations[("runtime", step)],
             "events": events_for_step}
            for step, events_for_step in sorted(grouped["runtime"].items())
        ],
    }
    return {
        "schema_version": 1,
        "lane": "QWEN3 CALL STACK",
        "phases": phases,
    }


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
