from scripts.analyze_async_timeline import analyze


def _event(name, ts, dur=1, cat="cpu"):
    return {"name": name, "ph": "X", "ts": ts, "dur": dur, "cat": cat}


def test_analyzer_accepts_submit_before_previous_sample():
    trace = {"traceEvents": [
        _event("auto_infer.async/0/submit", 1, 1),
        _event("auto_infer.async/1/submit", 3, 1),
        _event("ArgMax", 8, 1, "npu_kernel"),
        _event("ArgMax", 18, 1, "npu_kernel"),
    ]}

    verdict = analyze(trace)

    assert verdict["markers_present"]
    assert verdict["ordering_pass"]
    assert verdict["clone_free"]


def test_analyzer_rejects_host_bubble_and_clone():
    trace = {"traceEvents": [
        _event("auto_infer.async/0/submit", 1, 1),
        _event("auto_infer.async/1/submit", 12, 1),
        _event("ArgMax", 8, 1, "npu_kernel"),
        _event("ArgMax", 18, 1, "npu_kernel"),
        _event("aten::clone", 4, 1),
    ]}

    verdict = analyze(trace)

    assert not verdict["ordering_pass"]
    assert not verdict["clone_free"]


def test_analyzer_requires_both_markers_and_device_samples():
    verdict = analyze({"traceEvents": [_event("ArgMax", 8, 1)]})

    assert not verdict["markers_present"]
    assert not verdict["ordering_pass"]
