"""Single composition root for production execution backends."""

import json
import os

from auto_infer.config import EngineConfig
from auto_infer.executor_backends import get_executor_backend


def load_model(model_path: str, device_index: int, dtype: str):
    """Load one registered model; executor wrappers share this boundary."""
    from auto_infer.distributed import parallel_state
    from auto_infer.models.registry import get_model_class
    from auto_infer.platform import default_dtype, npu_device
    with open(os.path.join(model_path, "config.json")) as config_file:
        architecture = json.load(config_file)["architectures"][0]
    model_class = get_model_class(architecture)
    tp_size = parallel_state.tp_size()
    parallel_kwargs = {
        "ep_size": parallel_state.ep_size(),
        "ep_rank": parallel_state.ep_rank(),
    }
    if tp_size > 1:
        if not getattr(model_class, "SUPPORTS_TENSOR_PARALLEL", False):
            raise ValueError(
                f"{architecture} does not support tensor parallel execution")
        parallel_kwargs.update(
            tp_size=tp_size,
            tp_rank=parallel_state.tp_rank(),
        )
    return model_class.from_pretrained(
        model_path, device=npu_device(device_index), dtype=default_dtype(dtype),
        **parallel_kwargs)


def executor_arguments(config: EngineConfig) -> tuple[str, dict]:
    """Return the selected backend and constructor args derived from EngineConfig."""
    backend = get_executor_backend(config.execution.mode)
    backend.validate(config)
    return config.execution.mode, backend.arguments(config)


def build_executor(config: EngineConfig):
    mode, kwargs = executor_arguments(config)
    if config.model.model_package is not None:
        from auto_infer.models.registry import register_package
        register_package(
            config.model.model_package, config.model.model_path)
    from auto_infer.distributed.parallel_state import init_distributed
    init_distributed(config.parallel)
    return get_executor_backend(mode).load()(**kwargs)
