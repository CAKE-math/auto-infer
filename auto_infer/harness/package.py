"""Generate and validate the single runtime model-package descriptor."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re

from auto_infer.harness.artifacts import sha256_file


PACKAGE_FILE = "model-package.json"


class PackageValidationError(ValueError):
    pass


def _package_path(package_dir: Path) -> Path:
    path = package_dir.resolve()
    return path / PACKAGE_FILE if path.is_dir() else path


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "generated-model"


def generate_package(manifest: dict, output_dir: Path) -> dict:
    capability = manifest.get("capability")
    if not isinstance(capability, dict):
        raise PackageValidationError("manifest has no capability verdict")
    if capability.get("status") != "supported":
        missing = ", ".join(capability.get("missing", ()))
        raise PackageValidationError(
            f"model capability is not supported: {missing}")
    source = manifest["source"]
    architecture = source.get("architecture")
    if not architecture:
        raise PackageValidationError("model architecture is missing")
    package = {
        "schema_version": 1,
        "name": _slug(source.get("model_type") or architecture),
        "architectures": [architecture],
        "source": {
            "config_sha256": source["config_sha256"],
            "model_type": source.get("model_type"),
        },
        "implementation": {
            "template": capability["template"],
            "entrypoint": capability["entrypoint"],
        },
        "execution": {
            "attention": manifest["architecture"]["attention"],
            "dtype": "bfloat16",
            "quantization": {
                "enabled": False,
                "interface": "reserved",
            },
        },
        "features": dict(manifest["features"]),
        "validation": {
            "static": "pending",
            "runtime": "pending",
        },
    }
    _write_json(package, output_dir / PACKAGE_FILE)
    return package


def _require_dict(payload: dict, key: str) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PackageValidationError(f"package field {key!r} must be an object")
    return value


def validate_package(package_dir: Path, model_path: Path) -> dict:
    path = _package_path(package_dir)
    if not path.is_file():
        raise PackageValidationError(f"missing {PACKAGE_FILE}: {path}")
    try:
        package = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PackageValidationError(f"invalid package JSON: {error}") from error
    if package.get("schema_version") != 1:
        raise PackageValidationError("unsupported model-package schema version")
    architectures = package.get("architectures")
    if (not isinstance(architectures, list) or not architectures
            or not all(isinstance(item, str) and item for item in architectures)):
        raise PackageValidationError(
            "package architectures must be a non-empty string list")
    source = _require_dict(package, "source")
    config_path = model_path.resolve() / "config.json"
    if not config_path.is_file():
        raise PackageValidationError(f"missing model config: {config_path}")
    if source.get("config_sha256") != sha256_file(config_path):
        raise PackageValidationError(
            "model config fingerprint does not match package source")
    config = json.loads(config_path.read_text())
    architecture = (config.get("architectures") or [None])[0]
    if architecture not in architectures:
        raise PackageValidationError(
            f"checkpoint architecture {architecture!r} is not registered by package")
    implementation = _require_dict(package, "implementation")
    entrypoint = implementation.get("entrypoint")
    if not isinstance(entrypoint, str) or ":" not in entrypoint:
        raise PackageValidationError(
            "package implementation.entrypoint must be module-or-file:Class")
    if entrypoint.startswith("./"):
        relative, _ = entrypoint.split(":", 1)
        if not (path.parent / relative).is_file():
            raise PackageValidationError(
                f"relative package entrypoint does not exist: {relative}")
    execution = _require_dict(package, "execution")
    if execution.get("dtype") != "bfloat16":
        raise PackageValidationError("generated model packages must use bfloat16")
    quant = _require_dict(execution, "quantization")
    if quant.get("enabled") is not False or quant.get("interface") != "reserved":
        raise PackageValidationError(
            "quantization must be disabled with its interface reserved")
    return package


def load_entrypoint(package_dir: Path, entrypoint: str):
    target, class_name = entrypoint.rsplit(":", 1)
    if target.startswith("./"):
        file_path = (package_dir.resolve() / target).resolve()
        try:
            file_path.relative_to(package_dir.resolve())
        except ValueError:
            raise PackageValidationError(
                "relative entrypoint escapes the model package") from None
        module_name = (
            "_auto_infer_model_package_"
            + hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
        )
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise PackageValidationError(
                f"cannot import package entrypoint: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        model_class = getattr(module, class_name)
    except AttributeError:
        raise PackageValidationError(
            f"entrypoint class does not exist: {entrypoint}") from None
    if not callable(getattr(model_class, "from_pretrained", None)):
        raise PackageValidationError(
            f"entrypoint lacks from_pretrained: {entrypoint}")
    if not callable(getattr(model_class, "forward", None)):
        raise PackageValidationError(f"entrypoint lacks forward: {entrypoint}")
    if not getattr(model_class, "ATTENTION_FAMILY", None):
        raise PackageValidationError(
            f"entrypoint lacks ATTENTION_FAMILY: {entrypoint}")
    return model_class

