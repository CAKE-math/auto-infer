from scripts.analyze_async_timeline import analyze
from auto_infer.engine.async_timeline import (
    add_async_events_to_trace, clear_async_events, record_async_interval)


def _event(name, ts, dur=1, cat="cpu"):
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "cat": cat}


def test_analyzer_accepts_submit_before_previous_sample():
    trace = {"traceEvents": [
        _event("auto_infer.async/0/submit", 1, 1),
        _event("auto_infer.async/1/submit", 3, 1),
        _event("auto_infer.async/2/submit", 12, 1),
        _event("auto_infer.async/0/task_update", 2, 1),
        _event("auto_infer.async/1/task_update", 4, 1),
        _event("auto_infer.async/2/task_update", 14, 1),
        _event("ArgMaxV2AiCore_ArgMaxV2", 8, 1, "npu_kernel"),
        _event("aclnnIndex_IndexAiCore_Index", 10, 1, "npu_kernel"),
        _event("ArgMaxV2AiCore_ArgMaxV2", 18, 1, "npu_kernel"),
        _event("aclnnIndex_IndexAiCore_Index", 20, 1, "npu_kernel"),
        _event("ArgMaxV2AiCore_ArgMaxV2", 28, 1, "npu_kernel"),
    ]}

    verdict = analyze(trace)

    assert verdict["markers_present"]
    assert verdict["ordering_pass"]
    assert verdict["clone_free"]
    assert verdict["device_gap_p50_us"] == 1


def test_analyzer_rejects_host_bubble_and_clone():
    trace = {"traceEvents": [
        _event("auto_infer.async/0/submit", 1, 1),
        _event("auto_infer.async/1/submit", 12, 1),
        _event("auto_infer.async/2/submit", 22, 1),
        _event("auto_infer.async/0/task_update", 2, 1),
        _event("auto_infer.async/1/task_update", 13, 1),
        _event("auto_infer.async/2/task_update", 24, 1),
        _event("ArgMaxV2AiCore_ArgMaxV2", 8, 1, "npu_kernel"),
        _event("ArgMaxV2AiCore_ArgMaxV2", 18, 1, "npu_kernel"),
        _event("ArgMaxV2AiCore_ArgMaxV2", 28, 1, "npu_kernel"),
        _event("aten::clone", 14, 1),
    ]}

    verdict = analyze(trace)

    assert not verdict["ordering_pass"]
    assert not verdict["clone_free"]


def test_analyzer_requires_both_markers_and_device_samples():
    verdict = analyze({"traceEvents": [
        _event("ArgMaxV2AiCore_ArgMaxV2", 8, 1)]})

    assert not verdict["markers_present"]
    assert not verdict["ordering_pass"]


def test_background_task_update_events_are_added_to_chrome_trace(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text('{"traceEvents":[]}')
    clear_async_events()
    record_async_interval(7, "task_update", 1_000_000, 1_500_000)

    add_async_events_to_trace(path)

    import json
    event = json.loads(path.read_text())["traceEvents"][0]
    assert event["name"] == "auto_infer.async/7/task_update"
    assert event["ts"] == 1000
    assert event["dur"] == 500
