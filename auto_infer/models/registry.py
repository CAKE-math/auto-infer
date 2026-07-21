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


register("Qwen2ForCausalLM", Qwen2Model)
register("Qwen3ForCausalLM", Qwen3Model)
register("DeepseekV2ForCausalLM", DeepseekV2Model)
register("DeepseekV3ForCausalLM", DeepseekV2Model)
register("MiMoForCausalLM", Qwen2Model)
