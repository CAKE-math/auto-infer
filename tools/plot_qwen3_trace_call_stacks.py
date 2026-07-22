"""Render a dependency-free SVG from measured Qwen3 runtime call ranges."""

import html
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "docs" / "profiling" / "qwen3" / "summary.json"
OUTPUT = ROOT / "figures" / "qwen3-trace-call-stack-comparison.svg"
FRAMEWORKS = ("auto-infer", "omni-npu", "vllm-ascend")
COLORS = {
    "engine": "#087e8b",
    "llm-engine": "#087e8b",
    "engine-client": "#6d7f91",
    "engine-core": "#3d6485",
    "scheduler": "#d99024",
    "executor": "#315f8c",
    "worker-wrapper": "#75869a",
    "worker": "#58718c",
    "runner": "#238b68",
    "model-runner": "#238b68",
    "submit": "#30a179",
    "eager": "#59ad75",
    "decode-graph": "#177252",
    "prefill-graph": "#177252",
}


def representative_decode(call_stack: dict) -> dict:
    steps = call_stack["phases"]["decode"]
    median = statistics.median(step["duration_us"] for step in steps)
    return min(steps, key=lambda step: (
        abs(step["duration_us"] - median), step["step"]))


def _short_symbol(symbol: str) -> str:
    parts = symbol.split(".")
    return ".".join(parts[-2:])


def _text(parts, x, y, value, css="label", anchor="start"):
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" '
        f'text-anchor="{anchor}">{html.escape(str(value))}</text>')


def _stack_panel(parts, x, y, width, events, phase_ms, scale_ms, label):
    panel_height = 250
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{panel_height}" '
        'rx="8" class="panel"/>')
    _text(parts, x + 16, y + 25, label, "panel-title")
    _text(parts, x + width - 16, y + 25, f"{phase_ms:.2f} ms",
          "panel-value", "end")
    label_width = 240
    timeline_x = x + label_width
    timeline_width = width - label_width - 16
    row_height = 25
    top = y + 42
    base_ts = min(event["timestamp_us"] for event in events)
    for index, event in enumerate(events):
        row_y = top + index * row_height
        if index % 2:
            parts.append(
                f'<rect x="{x + 8}" y="{row_y - 14}" '
                f'width="{width - 16}" height="{row_height}" class="stripe"/>')
        depth = int(event["depth"])
        symbol = _short_symbol(event["symbol"])
        _text(parts, x + 14 + depth * 7, row_y + 3,
              f"d{depth}  {symbol}", "call-label")
        offset_ms = (event["timestamp_us"] - base_ts) / 1000
        duration_ms = event["duration_us"] / 1000
        bar_x = timeline_x + offset_ms / scale_ms * timeline_width
        bar_width = max(duration_ms / scale_ms * timeline_width, 1.5)
        color = COLORS.get(event["layer"], "#7b8794")
        parts.append(
            f'<rect x="{bar_x:.1f}" y="{row_y - 11:.1f}" '
            f'width="{bar_width:.1f}" height="15" rx="2" fill="{color}">'
            f'<title>{html.escape(event["layer"])} · '
            f'{html.escape(event["symbol"])} · {duration_ms:.3f} ms</title></rect>')
    axis_y = y + panel_height - 18
    parts.append(
        f'<line x1="{timeline_x}" y1="{axis_y}" '
        f'x2="{timeline_x + timeline_width}" y2="{axis_y}" class="axis"/>')
    for fraction in (0, .5, 1):
        tick_x = timeline_x + timeline_width * fraction
        parts.append(
            f'<line x1="{tick_x}" y1="{axis_y - 3}" '
            f'x2="{tick_x}" y2="{axis_y + 3}" class="axis"/>')
        _text(parts, tick_x, axis_y + 14, f"{scale_ms * fraction:.0f}",
              "tick", "middle")


def build_svg(summary: dict) -> str:
    profiles = summary["profiles"]
    decode_steps = {
        framework: representative_decode(
            profiles[framework]["runtime_call_stack"])
        for framework in FRAMEWORKS
    }
    decode_medians = {
        framework: statistics.median(
            step["duration_us"] for step in profiles[framework][
                "runtime_call_stack"]["phases"]["decode"]) / 1000
        for framework in FRAMEWORKS
    }
    prefill_ms = {
        framework: profiles[framework]["execution_phases"]["steps"][0][
            "duration_us"] / 1000
        for framework in FRAMEWORKS
    }
    auto_decode = decode_medians["auto-infer"]
    max_prefill = max(prefill_ms.values())
    max_decode = max(
        step["duration_us"] / 1000 for step in decode_steps.values())
    width, height = 1600, 930
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Qwen3 trace-derived runtime call-stack comparison</title>',
        '<desc id="desc">Measured prefill and median decode host ranges for '
        'auto-infer, omni-npu, and vllm-ascend.</desc>',
        '<style>text{font-family:Inter,Arial,"PingFang SC",sans-serif;fill:#122033}'
        '.bg{fill:#f4f7fa}.panel{fill:#fff;stroke:#ced8e3;stroke-width:1}'
        '.stripe{fill:#f3f6f8}.eyebrow{font-size:13px;font-weight:700;letter-spacing:1.5px;fill:#087e8b}'
        '.headline{font-size:24px;font-weight:750}.framework{font-size:18px;font-weight:750}'
        '.metric{font-size:14px;font-weight:650}.note{font-size:13px;fill:#5b6878}'
        '.panel-title{font-size:13px;font-weight:700}.panel-value{font-size:15px;font-weight:750}'
        '.call-label{font:11px "SFMono-Regular",Consolas,monospace;fill:#34495e}'
        '.axis{stroke:#8796a5;stroke-width:1}.tick{font-size:10px;fill:#6a7785}'
        '.footer{font-size:12px;fill:#5b6878}</style>',
        f'<rect width="{width}" height="{height}" class="bg"/>',
    ]
    _text(parts, 28, 32, "TRACE-DERIVED HOST CALL STACK", "eyebrow")
    omni_speedup = decode_medians["omni-npu"] / auto_decode
    vllm_speedup = decode_medians["vllm-ascend"] / auto_decode
    _text(parts, 28, 67,
          f"Decode: auto-infer {omni_speedup:.2f}× faster than omni-npu, "
          f"{vllm_speedup:.2f}× faster than vllm-ascend",
          "headline")
    _text(parts, 28, 94,
          "Prefill is shown separately: vllm-ascend leads this captured window; "
          "stack depth alone is not treated as causality.", "note")

    column_width = 500
    column_x = (25, 550, 1075)
    for framework, x in zip(FRAMEWORKS, column_x):
        calls = decode_steps[framework]["events"]
        nesting = max(event["depth"] for event in calls) + 1
        _text(parts, x + column_width / 2, 132, framework,
              "framework", "middle")
        relative = decode_medians[framework] / auto_decode
        comparison = (
            "baseline" if framework == "auto-infer"
            else f"{relative:.2f}× slower")
        _text(parts, x + column_width / 2, 154,
              f"decode median {decode_medians[framework]:.2f} ms · "
              f"max nesting {nesting} · {comparison}", "metric", "middle")
        prefill_events = profiles[framework]["runtime_call_stack"][
            "phases"]["prefill"]
        _stack_panel(parts, x, 174, column_width, prefill_events,
                     prefill_ms[framework], max_prefill,
                     "PREFILL · measured runtime boundaries")
        _stack_panel(parts, x, 456, column_width, calls,
                     decode_steps[framework]["duration_us"] / 1000,
                     max_decode,
                     f'DECODE {decode_steps[framework]["step"]:03d} · '
                     "nearest-to-median step")

    _text(parts, 28, 744,
          "Common x-scale within each row (ms). Bar position and width come from "
          "qwen3/call ranges in the raw trace; dN is measured nesting depth.",
          "note")
    _text(parts, 28, 778,
          "Observed target scope: auto-infer does not traverse the instrumented "
          "vLLM client/core or worker-wrapper/worker boundaries; decode is shorter.",
          "metric")
    _text(parts, 28, 806,
          "What it does not prove: fewer Python boundaries alone caused the speedup. "
          "Device graph work is asynchronous and requires operator-level attribution.",
          "note")
    legend_y = 850
    legend = (("engine/core", "#087e8b"), ("scheduler", "#d99024"),
              ("executor", "#315f8c"), ("worker/runner", "#58718c"),
              ("graph/eager", "#177252"))
    cursor = 28
    for label, color in legend:
        parts.append(
            f'<rect x="{cursor}" y="{legend_y - 12}" width="14" height="14" '
            f'rx="2" fill="{color}"/>')
        _text(parts, cursor + 20, legend_y, label, "footer")
        cursor += 150
    _text(parts, 28, 894,
          "Source: docs/profiling/qwen3/raw/*.trace.json · B16 · BF16 · "
          "1 prefill + 15 continuous decode steps · profiler-instrumented host time",
          "footer")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    OUTPUT.write_text(build_svg(summary))


if __name__ == "__main__":
    main()
