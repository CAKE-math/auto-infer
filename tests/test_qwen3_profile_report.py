import json

import pytest

from benchmarks.qwen3_profile_common import (
    sha256_file,
    validate_chrome_trace,
    write_profile_metadata,
)


def test_validate_chrome_trace_accepts_trace_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "GraphReplay", "ph": "X", "ts": 10, "dur": 4,
         "pid": 1, "tid": 2}
    ]}))

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
