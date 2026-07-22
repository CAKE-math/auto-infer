import json

import pytest

from benchmarks.qwen3_profile_common import (
    sha256_file,
    validate_chrome_trace,
    write_profile_metadata,
)
from benchmarks.profile_qwen3 import profile_configuration


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


@pytest.mark.parametrize(
    "framework", ["auto-infer", "omni-npu", "vllm-ascend"])
def test_profile_configuration_accepts_supported_frameworks(framework):
    assert profile_configuration(_manifest(), framework)["framework"] == framework


def test_profile_configuration_rejects_unknown_framework():
    with pytest.raises(ValueError, match="unsupported framework"):
        profile_configuration(_manifest(), "unknown")


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
