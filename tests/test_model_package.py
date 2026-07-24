import json

import pytest

from auto_infer.harness.capabilities import match_capabilities
from auto_infer.harness.inspect import inspect_model
from auto_infer.harness.package import (
    PackageValidationError,
    generate_package,
    validate_package,
)
from auto_infer.models.registry import get_model_class, register_package
from tests.test_harness_inspect import _gqa_config, _gqa_keys, _write_model


def test_generate_package_is_deterministic_and_bf16_only(tmp_path):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    manifest = inspect_model(model)
    manifest["capability"] = match_capabilities(manifest)
    output = tmp_path / "package"

    first = generate_package(manifest, output)
    second = generate_package(manifest, output)

    assert first == second
    assert json.loads((output / "model-package.json").read_text()) == first
    assert first["architectures"] == ["ExampleForCausalLM"]
    assert first["implementation"] == {
        "entrypoint": "auto_infer.models.qwen2:Qwen2Model",
        "template": "gqa-swiglu-v1",
    }
    assert first["execution"]["dtype"] == "bfloat16"
    assert first["execution"]["quantization"] == {
        "enabled": False,
        "interface": "reserved",
    }


def test_generate_package_rejects_partial_capability(tmp_path):
    model = _write_model(tmp_path, _gqa_config())
    manifest = inspect_model(model)
    manifest["capability"] = match_capabilities(manifest)

    with pytest.raises(PackageValidationError, match="not supported"):
        generate_package(manifest, tmp_path / "package")


def test_validate_package_rejects_checkpoint_fingerprint_drift(tmp_path):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    manifest = inspect_model(model)
    manifest["capability"] = match_capabilities(manifest)
    package_dir = tmp_path / "package"
    generate_package(manifest, package_dir)
    config = json.loads((model / "config.json").read_text())
    config["hidden_size"] = 2048
    (model / "config.json").write_text(json.dumps(config))

    with pytest.raises(PackageValidationError, match="fingerprint"):
        validate_package(package_dir, model)


def test_register_package_loads_relative_custom_entrypoint(tmp_path):
    model = _write_model(
        tmp_path,
        _gqa_config(
            model_type="custom_pkg",
            architectures=["CustomPackageForCausalLM"],
        ),
        _gqa_keys(),
    )
    package_dir = tmp_path / "custom-package"
    package_dir.mkdir()
    (package_dir / "model.py").write_text(
        "class CustomModel:\n"
        "    ATTENTION_FAMILY = 'gqa'\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path, device, dtype, **kwargs):\n"
        "        return cls()\n"
        "    def forward(self, ctx):\n"
        "        return ctx\n"
    )
    package = {
        "schema_version": 1,
        "name": "custom-package",
        "architectures": ["CustomPackageForCausalLM"],
        "source": {
            "config_sha256": inspect_model(model)["source"]["config_sha256"]
        },
        "implementation": {
            "template": "custom-v1",
            "entrypoint": "./model.py:CustomModel",
        },
        "execution": {
            "attention": "gqa",
            "dtype": "bfloat16",
            "quantization": {"enabled": False, "interface": "reserved"},
        },
        "validation": {"static": "pending", "runtime": "pending"},
    }
    (package_dir / "model-package.json").write_text(json.dumps(package))

    register_package(str(package_dir), str(model))

    assert get_model_class("CustomPackageForCausalLM").__name__ == "CustomModel"


def test_register_package_rejects_different_duplicate_mapping(tmp_path):
    model = _write_model(
        tmp_path,
        _gqa_config(
            model_type="duplicate_pkg",
            architectures=["Qwen3ForCausalLM"],
        ),
        _gqa_keys(),
    )
    package_dir = tmp_path / "duplicate-package"
    package_dir.mkdir()
    (package_dir / "model.py").write_text(
        "class OtherModel:\n"
        "    ATTENTION_FAMILY = 'gqa'\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, *args, **kwargs): return cls()\n"
        "    def forward(self, ctx): return ctx\n"
    )
    package = {
        "schema_version": 1,
        "name": "duplicate-package",
        "architectures": ["Qwen3ForCausalLM"],
        "source": {
            "config_sha256": inspect_model(model)["source"]["config_sha256"]
        },
        "implementation": {
            "template": "custom-v1",
            "entrypoint": "./model.py:OtherModel",
        },
        "execution": {
            "attention": "gqa",
            "dtype": "bfloat16",
            "quantization": {"enabled": False, "interface": "reserved"},
        },
        "validation": {"static": "pending", "runtime": "pending"},
    }
    (package_dir / "model-package.json").write_text(json.dumps(package))

    with pytest.raises(ValueError, match="already registered"):
        register_package(str(package_dir), str(model))
