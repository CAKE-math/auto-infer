import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from benchmarks.qwen3_profile_common import (
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
)
from tools.build_qwen3_architecture_report import build_report


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
        assert environment["driver"]["physical_npu"] == 1
    provenance = manifest["provenance"]
    assert provenance["model"]["prompt_token_count"] == len(
        provenance["model"]["prompt_token_ids"])
    assert provenance["model"]["config_sha256"]
    assert provenance["model"]["checkpoint_sha256"]
