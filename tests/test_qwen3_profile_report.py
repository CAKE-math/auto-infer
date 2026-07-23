import json
import importlib
import statistics
from contextlib import contextmanager
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from benchmarks import qwen3_profile_common
from benchmarks import profile_qwen3 as qwen3_profile
from tools import analyze_qwen3_profiles as qwen3_analyzer
from benchmarks.qwen3_profile_common import (
    StepPhaseRecorder,
    add_visible_phase_lane,
    extract_phase_index,
    sha256_file,
    validate_chrome_trace,
    write_profile_metadata,
)
from benchmarks.profile_qwen3 import (
    _revision,
    prepare_omni_compatibility,
    profile_configuration,
)
from tools.analyze_qwen3_profiles import (
    _publish_trace,
    classify_event,
    summarize_trace,
    validate_publishable_auto_profile,
)
from tools.build_qwen3_architecture_report import (
    build_markdown_report,
    build_report,
)


def _manifest():
    return {
        "model": "/data1/models/Qwen3-0.6B",
        "prompt": "Explain how a transformer decodes text.",
        "max_model_len": 2048,
        "output_tokens": 128,
        "throughput_batch": 16,
        "warmup_runs": 1,
        "measured_runs": 20,
        "usable_kv_tokens": 14464,
        "kv_block_size": 16,
        "kv_cache_memory_bytes": 1658847232,
        "async_scheduling": False,
        "async_batches": 2,
        "dtype": "bfloat16",
        "temperature": 0.0,
        "seed": 0,
    }


def test_profile_configuration_is_bounded_and_matched():
    config = profile_configuration(_manifest(), "auto-infer")

    assert config["batch_size"] == 16
    assert config["output_tokens"] == 16
    assert config["warmup_runs"] == 1
    assert config["usable_kv_tokens"] == 14464
    assert config["max_prefill_tokens"] == 256
    assert config["capture_phases"] == {
        "prefill_passes": 1,
        "decode_passes": 15,
        "continuous_decode": True,
        "speculative_mtp": False,
    }


@pytest.mark.parametrize(
    "framework", ["auto-infer", "omni-npu", "vllm-ascend"])
def test_profile_configuration_accepts_supported_frameworks(framework):
    assert profile_configuration(_manifest(), framework)["framework"] == framework


def test_profile_configuration_rejects_unknown_framework():
    with pytest.raises(ValueError, match="unsupported framework"):
        profile_configuration(_manifest(), "unknown")


def test_auto_profile_requires_prefill_graph_path():
    assert hasattr(qwen3_profile, "validate_auto_prefill_path")
    qwen3_profile.validate_auto_prefill_path({
        "phases": {"prefill": [{"layer": "prefill-graph"}]}})

    with pytest.raises(ValueError, match="prefill graph"):
        qwen3_profile.validate_auto_prefill_path({
            "phases": {"prefill": [{"layer": "eager"}]}})


def test_auto_profile_rejects_online_prefill_capture():
    valid = {
        "prefill_graph_steps": 1,
        "eager_steps": 0,
        "prefill_graph_online_captures": 0,
    }
    qwen3_profile.validate_auto_prefill_counters(valid)

    invalid = {**valid, "prefill_graph_online_captures": 1}
    with pytest.raises(ValueError, match="prefill graph counters"):
        qwen3_profile.validate_auto_prefill_counters(invalid)


def test_analyzer_revalidates_auto_prefill_before_publication():
    metadata = {"capture": {"profiled_path_counters": {
        "prefill_graph_steps": 1,
        "eager_steps": 0,
        "prefill_graph_online_captures": 0,
    }}}
    graph_path = {"phases": {"prefill": [{"layer": "prefill-graph"}]}}
    validate_publishable_auto_profile(metadata, graph_path)

    eager_path = {"phases": {"prefill": [{"layer": "eager"}]}}
    with pytest.raises(ValueError, match="prefill graph path"):
        validate_publishable_auto_profile(metadata, eager_path)

    metadata["capture"]["profiled_path_counters"][
        "prefill_graph_online_captures"] = 1
    with pytest.raises(ValueError, match="prefill graph counters"):
        validate_publishable_auto_profile(metadata, graph_path)


def test_analyzer_does_not_publish_before_auto_validation(
        tmp_path, monkeypatch):
    metadata_paths = []
    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        metadata = {
            "framework": framework,
            "workload": {
                "framework": framework,
                "output_tokens": 1,
                "capture_phases": {
                    "prefill_passes": 1,
                    "decode_passes": 0,
                    "continuous_decode": False,
                    "speculative_mtp": False,
                },
            },
            "environment": {"source_revision": "test"},
            "output_length": 1,
            "trace": {"sha256": "digest", "event_count": 1,
                      "file": f"{framework}.trace.json"},
            "capture": {"profiled_path_counters": {
                "prefill_graph_steps": 1,
                "eager_steps": 0,
                "prefill_graph_online_captures": (
                    1 if framework == "auto-infer" else 0),
            }},
        }
        path = tmp_path / f"{framework}.metadata.json"
        path.write_text(json.dumps(metadata))
        metadata_paths.append(path)

    source = tmp_path / "source.trace.json"
    source.write_text("[]")
    published = []
    monkeypatch.setattr(qwen3_analyzer, "load_comparable_results",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(qwen3_analyzer, "_locate_trace",
                        lambda *_args: source)
    monkeypatch.setattr(qwen3_analyzer, "validate_chrome_trace",
                        lambda _path: {"event_count": 1, "size_bytes": 2})
    monkeypatch.setattr(qwen3_analyzer, "sha256_file",
                        lambda _path: "digest")
    monkeypatch.setattr(qwen3_analyzer, "extract_call_stack_index",
                        lambda _path: {"phases": {"prefill": [
                            {"layer": "prefill-graph"}]}})
    monkeypatch.setattr(qwen3_analyzer, "_publish_trace",
                        lambda *args: published.append(args) or source)

    with pytest.raises(ValueError, match="prefill graph counters"):
        qwen3_analyzer.build_evidence(
            metadata_paths, [], tmp_path / "published")

    assert published == []


def test_prepare_omni_compatibility_supplies_model_slot(monkeypatch):
    monkeypatch.setattr("sys.argv", ["profile_qwen3.py", "manifest.json",
                                     "omni-npu", "output"])

    prepare_omni_compatibility("/models/qwen3")

    assert __import__("sys").argv[2] == "/models/qwen3"


def test_revision_prefers_explicit_deployed_source(monkeypatch):
    monkeypatch.setenv("AUTO_INFER_SOURCE_REVISION", "verified-revision")

    assert _revision() == "verified-revision"


def test_validate_chrome_trace_accepts_trace_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "GraphReplay", "ph": "X", "ts": 10, "dur": 4,
         "pid": 1, "tid": 2}
    ]}))

    result = validate_chrome_trace(path)

    assert result == {"event_count": 1, "size_bytes": path.stat().st_size}


def test_validate_chrome_trace_accepts_torch_npu_event_array(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        {"name": "npu_add", "ph": "X", "ts": "10.5", "dur": 4.0,
         "pid": 1, "tid": 2, "cat": "Ascend Hardware"}
    ]))

    result = validate_chrome_trace(path)

    assert result == {"event_count": 1, "size_bytes": path.stat().st_size}


def test_validate_chrome_trace_rejects_missing_event_array(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("{}")

    with pytest.raises(ValueError, match="traceEvents"):
        validate_chrome_trace(path)


def test_sha256_file_hashes_binary_content(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"profile")

    assert sha256_file(path) == (
        "1900eab6c028483d7126599ee6f50de0d27907b5c65fa90524580b4b0f9852b0")


def test_write_profile_metadata_is_canonical_and_validated(tmp_path):
    path = tmp_path / "nested" / "metadata.json"
    metadata = {
        "framework": "auto-infer",
        "trace": {"path": "auto-infer.trace.json"},
        "workload": {"batch_size": 16},
        "environment": {"device": "Ascend 910B1"},
        "output_digest": "digest",
        "output_length": 16,
    }

    write_profile_metadata(path, metadata)

    assert path.read_text().endswith("\n")
    assert json.loads(path.read_text()) == metadata
    assert path.read_text().index('"environment"') < path.read_text().index(
        '"framework"')


def test_write_profile_metadata_rejects_incomplete_payload(tmp_path):
    with pytest.raises(ValueError, match="output_length"):
        write_profile_metadata(tmp_path / "metadata.json", {
            "framework": "auto-infer",
            "trace": {},
            "workload": {},
            "environment": {},
            "output_digest": "digest",
        })


def test_step_phase_recorder_marks_prefill_then_each_decode_step():
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    class Engine:
        def step(self):
            return "token"

    engine = Engine()
    recorder = StepPhaseRecorder(record)

    with recorder.instrument(engine, output_tokens=4):
        for _ in range(4):
            assert engine.step() == "token"

    assert names == [
        "qwen3/phase/prefill",
        "qwen3/phase/decode/001",
        "qwen3/phase/decode/002",
        "qwen3/phase/decode/003",
    ]
    assert recorder.validate(output_tokens=4) == {
        "prefill_passes": 1,
        "decode_passes": 3,
        "runtime_drains": 0,
    }


def test_step_phase_recorder_keeps_terminal_runtime_drain_out_of_decode():
    names = []

    @contextmanager
    def record(name):
        names.append(name)
        yield

    class Engine:
        def step(self):
            return None

    engine = Engine()
    recorder = StepPhaseRecorder(record)

    with recorder.instrument(engine, output_tokens=3):
        for _ in range(4):
            engine.step()

    assert names == [
        "qwen3/phase/prefill",
        "qwen3/phase/decode/001",
        "qwen3/phase/decode/002",
        "qwen3/runtime/drain/001",
    ]
    assert recorder.validate(output_tokens=3)["runtime_drains"] == 1


def test_runtime_call_stack_recorder_wraps_real_nested_boundaries_and_restores():
    assert hasattr(qwen3_profile_common, "CallTarget")
    assert hasattr(qwen3_profile_common, "RuntimeCallStackRecorder")
    CallTarget = qwen3_profile_common.CallTarget
    RuntimeCallStackRecorder = qwen3_profile_common.RuntimeCallStackRecorder
    calls = []

    class Range:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            calls.append(("enter", self.name))

        def __exit__(self, *_):
            calls.append(("exit", self.name))

    class Scheduler:
        def schedule(self):
            return "batch"

    class Runner:
        def execute(self, batch):
            return batch

    class Engine:
        def __init__(self):
            self.scheduler = Scheduler()
            self.runner = Runner()

        def step(self):
            return self.runner.execute(self.scheduler.schedule())

    engine = Engine()
    original_step = engine.step
    recorder = RuntimeCallStackRecorder(Range)
    targets = (
        CallTarget("engine", engine, "step"),
        CallTarget("scheduler", engine.scheduler, "schedule"),
        CallTarget("runner", engine.runner, "execute"),
    )
    with recorder.instrument(targets):
        assert engine.step() == "batch"

    entered = [name for action, name in calls if action == "enter"]
    assert [name.split("/")[2] for name in entered] == [
        "engine", "scheduler", "runner"]
    assert all(name.startswith("qwen3/call/") for name in entered)
    assert recorder.counts == {"engine": 1, "scheduler": 1, "runner": 1}
    assert engine.step == original_step


def test_visible_call_stack_lane_is_trace_derived_and_preserves_nesting(tmp_path):
    assert hasattr(qwen3_profile_common, "add_visible_call_stack_lane")
    add_visible_call_stack_lane = qwen3_profile_common.add_visible_call_stack_lane
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        {"name": "qwen3/phase/prefill", "cat": "cpu_op", "ph": "X",
         "ts": 0, "dur": 100, "pid": 7, "tid": 8},
        {"name": "qwen3/call/engine/pkg.Engine.step", "cat": "cpu_op",
         "ph": "X", "ts": 1, "dur": 90, "pid": 7, "tid": 8},
        {"name": "qwen3/call/scheduler/pkg.Scheduler.schedule",
         "cat": "cpu_op", "ph": "X", "ts": 2, "dur": 10,
         "pid": 7, "tid": 8},
        {"name": "qwen3/call/executor/pkg.Executor.execute",
         "cat": "cpu_op", "ph": "X", "ts": 20, "dur": 70,
         "pid": 7, "tid": 8},
        {"name": "qwen3/call/runner/pkg.Runner.execute", "cat": "cpu_op",
         "ph": "X", "ts": 21, "dur": 60, "pid": 7, "tid": 8},
        {"name": "qwen3/phase/decode/001", "cat": "cpu_op", "ph": "X",
         "ts": 110, "dur": 50, "pid": 7, "tid": 8},
        {"name": "qwen3/call/engine/pkg.Engine.step", "cat": "cpu_op",
         "ph": "X", "ts": 111, "dur": 48, "pid": 7, "tid": 8},
    ]))

    index = add_visible_call_stack_lane(path)
    events = json.loads(path.read_text())

    assert any(event.get("args", {}).get("name") == "QWEN3 CALL STACK"
               for event in events)
    copied = [event for event in events if event.get("cat") == "qwen3.callstack"]
    assert len(copied) == 5
    assert [item["depth"] for item in index["phases"]["prefill"]] == [
        0, 1, 1, 2]
    assert index["phases"]["decode"][0]["step"] == 1
    assert index["phases"]["decode"][0]["events"][0]["symbol"] == (
        "pkg.Engine.step")
    assert hasattr(qwen3_profile_common, "extract_call_stack_index")
    assert qwen3_profile_common.extract_call_stack_index(path) == index


def test_framework_call_targets_follow_live_runtime_objects():
    assert hasattr(qwen3_profile, "_auto_call_targets")
    assert hasattr(qwen3_profile, "_vllm_call_targets")
    _auto_call_targets = qwen3_profile._auto_call_targets
    _vllm_call_targets = qwen3_profile._vllm_call_targets
    class Node:
        def step(self): pass
        def step_fn(self): pass
        def schedule(self): pass
        def execute(self): pass
        def execute_model(self): pass
        def get_output(self): pass
        def prepare(self): pass
        def submit(self): pass
        def submit_prepared(self): pass
        def _eager_submit(self): pass
        def _prepare_graph(self): pass
        def _submit_graph(self): pass
        def _prefill_graph_submit(self): pass

    auto = Node()
    auto.scheduler = Node()
    auto.executor = Node()
    auto.executor.runner = Node()
    assert [target.layer for target in _auto_call_targets(auto)] == [
        "engine", "scheduler", "executor", "runner", "prepare", "submit",
        "submit-prepared", "eager", "decode-graph-prepare",
        "decode-graph-submit", "prefill-graph"]

    llm_engine = Node()
    client = llm_engine.engine_core = Node()
    core = client.engine_core = Node()
    core.scheduler = Node()
    executor = core.model_executor = Node()
    wrapper = executor.driver_worker = Node()
    worker = wrapper.worker = Node()
    worker.model_runner = Node()
    assert [target.layer for target in _vllm_call_targets(llm_engine)] == [
        "llm-engine", "engine-client", "engine-core", "scheduler",
        "executor", "worker-wrapper", "worker", "model-runner"]


def test_visible_phase_lane_is_explicit_and_machine_validated(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        _event("qwen3/phase/prefill", 8, ts=100, pid=4, tid=9),
        _event("qwen3/phase/decode/001", 3, ts=110, pid=4, tid=9),
        _event("qwen3/phase/decode/002", 2, ts=114, pid=4, tid=9),
        _event("native_framework_event", 1, ts=111, pid=8, tid=3),
    ]))

    phase_index = add_visible_phase_lane(path, output_tokens=3)

    assert [item["phase"] for item in phase_index["steps"]] == [
        "prefill", "decode", "decode"]
    assert [item["step"] for item in phase_index["steps"]] == [0, 1, 2]
    assert extract_phase_index(path) == phase_index
    events = json.loads(path.read_text())
    assert any(event.get("args", {}).get("name") == "QWEN3 PHASES"
               for event in events)
    assert any(event.get("name") == "PREFILL" and
               event.get("cat") == "qwen3.phase" for event in events)
    assert any(event.get("name") == "DECODE 002" and
               event.get("cat") == "qwen3.phase" for event in events)


def test_visible_phase_lane_rejects_missing_decode_step(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        _event("qwen3/phase/prefill", 8, ts=100),
        _event("qwen3/phase/decode/001", 3, ts=110),
    ]))

    with pytest.raises(ValueError, match="phase marker mismatch"):
        add_visible_phase_lane(path, output_tokens=3)


@pytest.mark.parametrize(("name", "phase"), [
    ("GraphReplay", "graph_replay"),
    ("npu_fused_infer_attention_score_v2", "attention_kv"),
    ("npu_scatter_pa_kv_cache", "attention_kv"),
    ("npu_grouped_matmul", "projection_mlp_norm"),
    ("aten::argmax", "lm_head_sampling"),
    ("unknown_vendor_op", "unclassified"),
])
def test_classify_event(name, phase):
    assert classify_event(name) == phase


def _event(name, duration, **extra):
    return {
        "name": name,
        "ph": "X",
        "ts": 10,
        "dur": duration,
        "pid": 1,
        "tid": 2,
        **extra,
    }


def test_trace_summary_aggregates_complete_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        _event("GraphReplay", 5),
        _event("GraphReplay", 7),
        _event("unknown_vendor_op", 3),
        {"name": "metadata", "ph": "M", "pid": 1, "tid": 2},
    ]}))

    summary = summarize_trace(path)

    assert summary["event_count"] == 4
    assert summary["complete_event_count"] == 3
    assert summary["total_event_duration_us"] == 15
    assert summary["phases"]["graph_replay"] == {
        "count": 2, "duration_us": 12.0, "share": 0.8}
    assert summary["phases"]["unclassified"]["duration_us"] == 3
    assert summary["unclassified_names"] == ["unknown_vendor_op"]
    assert summary["top_events"][0]["name"] == "GraphReplay"


def test_trace_summary_accepts_numeric_string_duration_and_retains_identity(
        tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        _event("aten::argmax", "2.5", cat="cpu_op", pid=4, tid=5),
        _event("bad", "not-a-number"),
    ]))

    summary = summarize_trace(path)

    assert summary["complete_event_count"] == 1
    assert summary["top_events"][0]["categories"] == ["cpu_op"]
    assert summary["top_events"][0]["pids"] == [4]
    assert summary["top_events"][0]["tids"] == [5]


def test_trace_summary_excludes_runtime_call_stack_annotation_ranges(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([
        _event("GraphReplay", 7),
        _event("qwen3/call/engine/pkg.Engine.step", 6, cat="cpu_op"),
        _event("engine · pkg.Engine.step", 6, cat="qwen3.callstack"),
        _event("qwen3/runtime/drain/001", 2, cat="cpu_op"),
    ]))

    summary = summarize_trace(path)

    assert summary["complete_event_count"] == 1
    assert [event["name"] for event in summary["top_events"]] == [
        "GraphReplay"]


def test_publish_trace_copies_validated_artifact_to_raw_directory(tmp_path):
    source = tmp_path / "capture" / "trace.json"
    source.parent.mkdir()
    source.write_text(json.dumps([_event("GraphReplay", 1)]))

    published = _publish_trace(source, tmp_path / "report", "auto-infer")

    assert published == tmp_path / "report" / "raw" / "auto-infer.trace.json"
    assert published.read_bytes() == source.read_bytes()


@pytest.fixture
def profile_evidence():
    root = Path(__file__).parents[1] / "docs" / "profiling" / "qwen3"
    return (
        json.loads((root / "summary.json").read_text()),
        json.loads((root / "manifest.json").read_text()),
    )


def test_report_contains_executive_and_engineering_sections(profile_evidence):
    html = build_report(*profile_evidence)

    for anchor in (
        "executive-summary", "matched-benchmark", "profiling-deep-dive",
        "why-faster", "architecture-comparison", "invariants",
        "per-model-artifacts", "acceptance-workflow", "evidence-appendix",
    ):
        assert f'id="{anchor}"' in html


def test_report_links_every_raw_trace_and_labels_evidence(profile_evidence):
    html = build_report(*profile_evidence)

    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        assert f'profiling/qwen3/raw/{framework}.trace.json' in html
    assert "实测" in html
    assert "源码观察" in html
    assert "因果推断" in html
    assert "1 次 prefill" in html
    assert "15 次连续 decode" in html
    assert "不是投机 MTP" in html


def test_published_traces_contain_measured_runtime_call_stack_lanes(
        profile_evidence):
    summary, manifest = profile_evidence
    root = Path(__file__).parents[1] / "docs" / "profiling" / "qwen3"
    expected_boundaries = {
        "auto-infer": 6, "omni-npu": 8, "vllm-ascend": 8}

    for framework, expected in expected_boundaries.items():
        path = root / manifest["artifacts"][framework]["path"]
        events = qwen3_profile_common.load_chrome_trace_events(path)
        assert any(
            event.get("args", {}).get("name") == "QWEN3 CALL STACK"
            for event in events)
        source = [event for event in events if str(event.get("name", ""))
                  .startswith("qwen3/call/")]
        copied = [event for event in events
                  if event.get("cat") == "qwen3.callstack"]
        assert source
        assert len(copied) == len(source)
        call_stack = summary["profiles"][framework]["runtime_call_stack"]
        assert len(call_stack["phases"]["decode"]) == 15
        assert len(call_stack["phases"]["decode"][0]["events"]) == expected


def test_reports_include_call_stack_and_architecture_visuals(profile_evidence):
    summary, manifest = profile_evidence
    html = build_report(summary, manifest)
    markdown = build_markdown_report(summary, manifest)
    diagrams = (
        "qwen3-trace-call-stack-comparison.svg",
        "qwen3-three-framework-architecture.png",
        "qwen3-profile-phase-sequence.png",
    )

    for diagram in diagrams:
        assert f"../figures/{diagram}" in html
        assert f"../figures/{diagram}" in markdown
    for symbol in (
        "EngineCore.step", "GraphPagedRunner._graph_submit",
        "InprocClient.get_output", "NPUModelRunner.execute_model",
    ):
        assert symbol in html
        assert symbol in markdown
    auto_decode = statistics.median(
        step["duration_us"] for step in summary["profiles"]["auto-infer"][
            "runtime_call_stack"]["phases"]["decode"])
    vllm_decode = statistics.median(
        step["duration_us"] for step in summary["profiles"]["vllm-ascend"][
            "runtime_call_stack"]["phases"]["decode"])
    for fact in ("TRACE-DERIVED", "QWEN3 CALL STACK",
                 f"{auto_decode / 1000:.2f} ms",
                 f"{vllm_decode / auto_decode:.2f}× slower"):
        assert fact in html
        assert fact in markdown
    assert "prefill-graph" in html
    assert "prefill-graph" in markdown
    assert "prefill 则如实显示 vllm-ascend 领先" not in html


def test_management_conclusion_reports_memory_tradeoff(profile_evidence):
    summary, manifest = profile_evidence
    html = build_report(summary, manifest)
    markdown = build_markdown_report(summary, manifest)
    data = summary["headline_benchmarks"]

    assert data["auto-infer"]["peak_allocated_gib"] > data[
        "vllm-ascend"]["peak_allocated_gib"]
    for report in (html, markdown):
        assert "等容量内存与稳定性上全部第一" not in report
        assert "5.24 GiB" in report
        assert "2.80 GiB" in report


def test_html_print_layout_collapses_appendix_columns(profile_evidence):
    summary, manifest = profile_evidence
    html = build_report(summary, manifest)

    assert "@media print" in html
    assert ".appendix-grid{grid-template-columns:minmax(0,1fr)}" in html
    assert "white-space:pre-wrap" in html


def test_prefill_description_is_derived_from_provenance(profile_evidence):
    summary, manifest = deepcopy(profile_evidence)
    manifest["provenance"]["model"]["prompt_token_count"] = 7

    for report in (build_report(summary, manifest),
                   build_markdown_report(summary, manifest)):
        assert "B16 × 7-token prompt" in report
        assert "112 个 flattened query tokens" in report
        assert "B16 × 9-token prompt" not in report


def test_trace_call_stack_figure_is_generated_from_profile_summary(
        profile_evidence):
    try:
        plotter = importlib.import_module("tools.plot_qwen3_trace_call_stacks")
    except ModuleNotFoundError:
        plotter = None
    assert plotter is not None
    summary, _ = profile_evidence

    representative = plotter.representative_decode(
        summary["profiles"]["auto-infer"]["runtime_call_stack"])
    svg = plotter.build_svg(summary)

    durations = [
        step["duration_us"] for step in summary["profiles"]["auto-infer"][
            "runtime_call_stack"]["phases"]["decode"]]
    median = statistics.median(durations)
    assert abs(representative["duration_us"] - median) == min(
        abs(duration - median) for duration in durations)
    assert svg.startswith("<svg")
    assert "TRACE-DERIVED HOST CALL STACK" in svg
    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        assert framework in svg
    assert "GraphPagedRunner._graph_submit" in svg
    assert "NPUModelRunner.execute_model" in svg
    assert f"{median / 1000:.2f} ms" in svg
    vllm_median = statistics.median(
        step["duration_us"] for step in summary["profiles"]["vllm-ascend"][
            "runtime_call_stack"]["phases"]["decode"])
    assert f"{vllm_median / median:.2f}× slower" in svg


def test_markdown_report_is_full_and_uses_same_evidence(profile_evidence):
    summary, manifest = profile_evidence

    document = build_markdown_report(summary, manifest)

    assert document.startswith("# auto-infer 架构与 Qwen3 性能审计报告\n")
    for heading in (
        "管理结论", "Matched benchmark", "Qwen3 三框架 profiling",
        "为什么 auto-infer 更快", "架构优劣详细对比",
        "什么不应该变化", "新模型生产验收流程", "证据附录",
    ):
        assert heading in document
    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        assert f"profiling/qwen3/raw/{framework}.trace.json" in document
        assert manifest["artifacts"][framework]["sha256"] in document
    assert "PREFILL" in document
    assert "DECODE 001" in document


def test_report_is_offline_semantic_and_has_no_placeholders(profile_evidence):
    html = build_report(*profile_evidence)

    assert html.count("<title>") == 1
    assert html.count("<h1") == 1
    assert "<script" not in html
    assert "https://" not in html
    assert "http://" not in html
    assert "TODO" not in html
    assert "PLACEHOLDER" not in html
    assert "overflow-x:hidden" not in html
    assert "line-break:anywhere" in html


def test_report_headline_and_attention_facts_are_json_driven(profile_evidence):
    summary, manifest = deepcopy(profile_evidence)
    summary["headline_benchmarks"]["auto-infer"][
        "throughput_tokens_per_second"]["median"] = 999.0
    for benchmark in summary["headline_benchmarks"].values():
        benchmark["manifest"]["throughput_batch"] = 7
        benchmark["manifest"]["output_tokens"] = 64
    manifest["workload"]["batch_size"] = 3
    manifest["workload"]["output_tokens"] = 5
    manifest["workload"]["capture_phases"]["decode_passes"] = 4
    for event in summary["profiles"]["auto-infer"]["top_events"]:
        if event["name"].startswith("npu::npu_fused_infer_attention_score"):
            event["count"] = 123
    for share, profile in zip(
            (0.01, 0.02, 0.03), summary["profiles"].values()):
        profile["phases"]["unclassified"]["share"] = share

    document = build_report(summary, manifest)

    assert "B7: 999.0 tok/s" in document
    assert "B3 5-token 请求范围" in document
    assert "<strong>123</strong><span>FIA host calls</span>" in document
    assert "1.0%–3.0%" in document


class _ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.hrefs.append(attributes["href"])


def test_report_links_resolve_and_hashes_match(profile_evidence):
    summary, manifest = profile_evidence
    document = build_report(summary, manifest)
    parser = _ReportParser()
    parser.feed(document)
    docs = Path(__file__).parents[1] / "docs"

    for href in parser.hrefs:
        if href.startswith("#"):
            assert href[1:] in parser.ids
        else:
            assert (docs / href).is_file(), href
    for artifact in manifest["artifacts"].values():
        assert artifact["sha256"] in document
        environment = artifact["metadata"]["environment"]
        assert environment["capture_harness_revision"] != "unknown"
        assert environment["capture_harness_revision_origin"] in {
            "capture_environment", "content_hash_verified_deployment"}
        assert environment["framework_source_revisions"]
        assert environment["driver"]["physical_npu"] == manifest[
            "provenance"]["driver"]["physical_npu"]
    provenance = manifest["provenance"]
    assert provenance["model"]["prompt_token_count"] == len(
        provenance["model"]["prompt_token_ids"])
    assert provenance["model"]["config_sha256"]
    assert provenance["model"]["checkpoint_sha256"]
    assert provenance["driver"]["physical_npu"] == 4
    path_counters = manifest["artifacts"]["auto-infer"]["metadata"][
        "capture"]["profiled_path_counters"]
    assert path_counters["prefill_graph_steps"] == 1
    assert path_counters["eager_steps"] == 0
    assert path_counters["prefill_graph_online_captures"] == 0
