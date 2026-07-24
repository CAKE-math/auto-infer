import json

from auto_infer.harness.cli import main
from tests.test_harness_inspect import _gqa_config, _gqa_keys, _write_model


def test_inferctl_capabilities_is_structured(capsys):
    code = main(["capabilities"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["step_id"] == "capabilities"
    assert payload["result"]["templates"] == [
        "gqa-qknorm-v1",
        "gqa-swiglu-v1",
        "mla-moe-v1",
    ]


def test_inferctl_inspect_writes_fixed_artifacts(tmp_path, capsys):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    artifacts = tmp_path / "artifacts"
    checkpoint = tmp_path / "checkpoint.md"
    checkpoint.write_text("do-not-touch")

    code = main([
        "inspect", "model", str(model), "--artifacts", str(artifacts),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert (artifacts / "inspect-model.json").is_file()
    assert (artifacts / "model-manifest.json").is_file()
    assert checkpoint.read_text() == "do-not-touch"


def test_inferctl_adapt_generates_supported_package(tmp_path, capsys):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    artifacts = tmp_path / "artifacts"
    package = tmp_path / "package"

    code = main([
        "adapt", "model", str(model),
        "--output", str(package),
        "--artifacts", str(artifacts),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert (package / "model-package.json").is_file()
    assert (artifacts / "adapt-model.json").is_file()


def test_inferctl_adapt_returns_partial_for_unproven_weights(tmp_path, capsys):
    model = _write_model(tmp_path, _gqa_config())
    package = tmp_path / "package"

    code = main([
        "adapt", "model", str(model),
        "--output", str(package),
        "--artifacts", str(tmp_path / "artifacts"),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "partial"
    assert "weights.standard_layout" in payload["result"]["missing"]
    assert not package.exists()


def test_inferctl_validate_reports_fingerprint_failure(tmp_path, capsys):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    package = tmp_path / "package"
    assert main([
        "adapt", "model", str(model),
        "--output", str(package),
        "--artifacts", str(tmp_path / "adapt-artifacts"),
    ]) == 0
    capsys.readouterr()
    config = json.loads((model / "config.json").read_text())
    config["hidden_size"] = 2048
    (model / "config.json").write_text(json.dumps(config))

    code = main([
        "validate", "package", str(package), "--model", str(model),
        "--artifacts", str(tmp_path / "validate-artifacts"),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "failed"
    assert "fingerprint" in payload["error_summary"]
    assert (tmp_path / "validate-artifacts" /
            "validate-package.json").is_file()
