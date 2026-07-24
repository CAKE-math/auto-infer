"""Public non-interactive CLI for PIE and framework operators."""

import argparse
import json
from pathlib import Path

from auto_infer.harness.artifacts import (
    HarnessResult,
    build_provenance,
    exit_code,
    write_result,
)
from auto_infer.harness.capabilities import match_capabilities
from auto_infer.harness.inspect import inspect_model
from auto_infer.harness.package import (
    generate_package,
    load_entrypoint,
    validate_package,
)


_ARTIFACT_NAMES = {
    "inspect-model": "inspect-model.json",
    "adapt-model": "adapt-model.json",
    "validate-package": "validate-package.json",
}


def runtime_capabilities() -> dict:
    return {
        "schema_version": 1,
        "dtype": ["bfloat16"],
        "attention_families": ["gqa", "mla"],
        "templates": [
            "gqa-qknorm-v1",
            "gqa-swiglu-v1",
            "mla-moe-v1",
        ],
        "quantization": {
            "automatic": False,
            "interface": "reserved",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferctl")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "capabilities", help="print stable runtime capabilities")

    inspect = commands.add_parser("inspect")
    inspect_targets = inspect.add_subparsers(dest="target", required=True)
    inspect_model_parser = inspect_targets.add_parser("model")
    inspect_model_parser.add_argument("model")
    inspect_model_parser.add_argument(
        "--artifacts", default=".inferctl-artifacts")

    adapt = commands.add_parser("adapt")
    adapt_targets = adapt.add_subparsers(dest="target", required=True)
    adapt_model_parser = adapt_targets.add_parser("model")
    adapt_model_parser.add_argument("model")
    adapt_model_parser.add_argument("--output", required=True)
    adapt_model_parser.add_argument(
        "--artifacts", default=".inferctl-artifacts")

    validate = commands.add_parser("validate")
    validate_targets = validate.add_subparsers(dest="target", required=True)
    validate_package_parser = validate_targets.add_parser("package")
    validate_package_parser.add_argument("package")
    validate_package_parser.add_argument("--model", required=True)
    validate_package_parser.add_argument(
        "--artifacts", default=".inferctl-artifacts")
    return parser


def _print(result: HarnessResult) -> int:
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return exit_code(result.status)


def _persist(result: HarnessResult, artifacts_dir: Path) -> int:
    write_result(result, artifacts_dir / _ARTIFACT_NAMES[result.step_id])
    return _print(result)


def _inspect(model_path: Path, artifacts_dir: Path) -> int:
    manifest = inspect_model(model_path)
    manifest["capability"] = match_capabilities(manifest)
    manifest_path = artifacts_dir / "model-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = HarnessResult(
        status="ok",
        step_id="inspect-model",
        result={
            "architecture": manifest["architecture"],
            "capability": manifest["capability"],
        },
        artifacts={"model_manifest": str(manifest_path.resolve())},
        provenance=build_provenance(model_path / "config.json"),
    )
    return _persist(result, artifacts_dir)


def _adapt(model_path: Path, output_dir: Path,
           artifacts_dir: Path) -> int:
    manifest = inspect_model(model_path)
    capability = match_capabilities(manifest)
    manifest["capability"] = capability
    manifest_path = artifacts_dir / "model-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if capability["status"] != "supported":
        result = HarnessResult(
            status="partial",
            step_id="adapt-model",
            result={
                "missing": capability["missing"],
                "capability": capability,
            },
            artifacts={"model_manifest": str(manifest_path.resolve())},
            error_summary=(
                "model requires Agent adaptation: "
                + ", ".join(capability["missing"])
            ),
            provenance=build_provenance(model_path / "config.json"),
        )
        return _persist(result, artifacts_dir)
    package = generate_package(manifest, output_dir)
    package_path = output_dir / "model-package.json"
    result = HarnessResult(
        status="ok",
        step_id="adapt-model",
        result={"package": package, "missing": []},
        artifacts={
            "model_manifest": str(manifest_path.resolve()),
            "model_package": str(package_path.resolve()),
        },
        provenance=build_provenance(model_path / "config.json"),
    )
    return _persist(result, artifacts_dir)


def _validate(package_dir: Path, model_path: Path,
              artifacts_dir: Path) -> int:
    package = validate_package(package_dir, model_path)
    model_class = load_entrypoint(
        package_dir, package["implementation"]["entrypoint"])
    expected = package["execution"]["attention"]
    if expected in {"mha", "mqa"}:
        expected = "gqa"
    if model_class.ATTENTION_FAMILY != expected:
        raise ValueError(
            "model package attention mismatch: "
            f"{model_class.ATTENTION_FAMILY} != {expected}")
    result = HarnessResult(
        status="ok",
        step_id="validate-package",
        result={
            "architecture_count": len(package["architectures"]),
            "entrypoint": package["implementation"]["entrypoint"],
            "model_class": model_class.__name__,
        },
        artifacts={
            "model_package": str(
                (package_dir / "model-package.json").resolve())
        },
        provenance=build_provenance(model_path / "config.json"),
    )
    return _persist(result, artifacts_dir)


def _step_id(args) -> str:
    if args.command == "capabilities":
        return "capabilities"
    return f"{args.command}-{args.target}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capabilities":
        return _print(HarnessResult(
            status="ok",
            step_id="capabilities",
            result=runtime_capabilities(),
            provenance=build_provenance(),
        ))
    artifacts_dir = Path(args.artifacts)
    try:
        if args.command == "inspect":
            return _inspect(Path(args.model), artifacts_dir)
        if args.command == "adapt":
            return _adapt(
                Path(args.model), Path(args.output), artifacts_dir)
        return _validate(
            Path(args.package), Path(args.model), artifacts_dir)
    except Exception as error:
        result = HarnessResult(
            status="failed",
            step_id=_step_id(args),
            error_summary=str(error),
            provenance=build_provenance(),
        )
        return _persist(result, artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())

