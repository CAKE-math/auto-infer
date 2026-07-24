import json
import re

import pytest

from auto_infer.harness.artifacts import (
    HarnessResult,
    build_provenance,
    exit_code,
    sha256_file,
    write_result,
)


def test_harness_result_is_step_envelope_compatible(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"architectures":["ExampleForCausalLM"]}')
    result = HarnessResult(
        status="ok",
        step_id="inspect-model",
        result={"architecture": "ExampleForCausalLM"},
        artifacts={"model_manifest": "model-manifest.json"},
        provenance=build_provenance(config),
    )

    payload = result.to_dict()

    assert payload["status"] == "ok"
    assert payload["step_id"] == "inspect-model"
    assert payload["error_summary"] is None
    assert payload["artifacts"] == {
        "model_manifest": "model-manifest.json"
    }
    assert payload["provenance"]["config_sha256"] == sha256_file(config)
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
        payload["created_at"],
    )


def test_harness_result_writes_stable_sorted_json(tmp_path):
    path = tmp_path / "result.json"
    result = HarnessResult(
        status="partial",
        step_id="adapt-model",
        created_at="2026-07-24T00:00:00Z",
        result={"z": 1, "a": 2},
    )

    write_result(result, path)

    assert json.loads(path.read_text()) == result.to_dict()
    assert path.read_text().index('"a"') < path.read_text().index('"z"')
    assert path.read_text().endswith("\n")


@pytest.mark.parametrize(
    ("status", "expected"),
    [("ok", 0), ("partial", 2), ("failed", 1), ("skipped", 0)],
)
def test_harness_status_has_deterministic_exit_code(status, expected):
    assert exit_code(status) == expected


def test_harness_rejects_unknown_status():
    with pytest.raises(ValueError, match="unknown harness status"):
        exit_code("maybe")
