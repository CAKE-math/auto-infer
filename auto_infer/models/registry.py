"""HF architecture-name to auto-infer model-class registry."""
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.models.qwen2 import Qwen2Model
from auto_infer.models.qwen3 import Qwen3Model

_REGISTRY = {}


def get_model_class(architecture: str):
    if architecture not in _REGISTRY:
        raise KeyError(f"unregistered architecture: {architecture}")
    return _REGISTRY[architecture]


def register(architecture: str, cls) -> None:
    if not architecture:
        raise ValueError("model architecture must not be empty")
    if architecture in _REGISTRY:
        raise ValueError(f"model architecture already registered: {architecture}")
    _REGISTRY[architecture] = cls


def register_package(package_dir: str, model_path: str) -> None:
    """Register an explicit, fingerprint-checked generated model package."""
    from pathlib import Path
    from auto_infer.harness.package import (
        load_entrypoint,
        validate_package,
    )
    root = Path(package_dir)
    package = validate_package(root, Path(model_path))
    model_class = load_entrypoint(root, package["implementation"]["entrypoint"])
    expected_family = package["execution"]["attention"]
    if expected_family in {"mha", "mqa"}:
        expected_family = "gqa"
    if model_class.ATTENTION_FAMILY != expected_family:
        raise ValueError(
            "model package attention mismatch: "
            f"{model_class.ATTENTION_FAMILY} != {expected_family}")
    for architecture in package["architectures"]:
        existing = _REGISTRY.get(architecture)
        if existing is model_class:
            continue
        register(architecture, model_class)


register("Qwen2ForCausalLM", Qwen2Model)
register("Qwen3ForCausalLM", Qwen3Model)
register("DeepseekV2ForCausalLM", DeepseekV2Model)
register("DeepseekV3ForCausalLM", DeepseekV2Model)
register("MiMoForCausalLM", Qwen2Model)
