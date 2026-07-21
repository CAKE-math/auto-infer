import json

from benchmarks.run_serving_frontend import main as frontend_main
from benchmarks.run_serving_online import _git_revision, _request_body
from benchmarks.serving_common import validate_serving_result
from benchmarks.sse_client import parse_sse_event, parse_sse_line


def test_parse_sse_line_extracts_text_and_terminal():
    line = 'data: {"choices":[{"text":"hello","finish_reason":null}]}'

    assert parse_sse_line(line) == ("hello", False)
    assert parse_sse_line("data: [DONE]") == ("", True)
    assert parse_sse_line(": keepalive") is None

    counted = parse_sse_event(
        'data: {"completion_tokens":2,"choices":[{"text":"x"}]}'
    )
    assert counted is not None
    assert counted.completion_tokens == 2


def test_frontend_smoke_writes_schema_complete_result(tmp_path):
    output = tmp_path / "serving-smoke.json"

    assert frontend_main(["--smoke", "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    validate_serving_result(result)
    assert result["framework"] == "auto-infer-smoke"


def test_git_revision_accepts_explicit_revision_for_exported_source(monkeypatch):
    monkeypatch.setenv("AUTO_INFER_GIT_COMMIT", "exported-source-revision")

    assert _git_revision() == "exported-source-revision"


def test_online_benchmark_forces_exact_output_length():
    body = _request_body("prompt", 32, "model")

    assert body["ignore_eos"] is True
    assert body["max_tokens"] == 32
