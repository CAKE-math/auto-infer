"""Single composition root for production execution backends."""

import json
import os

from auto_infer.config import EngineConfig
from auto_infer.executor_backends import get_executor_backend


def load_model(model_path: str, device_index: int, dtype: str):
    """Load one registered model; executor wrappers share this boundary."""
    from auto_infer.distributed.parallel_state import ep_rank, ep_size
    from auto_infer.models.registry import get_model_class
    from auto_infer.platform import default_dtype, npu_device
    with open(os.path.join(model_path, "config.json")) as config_file:
        architecture = json.load(config_file)["architectures"][0]
    return get_model_class(architecture).from_pretrained(
        model_path, device=npu_device(device_index), dtype=default_dtype(dtype),
        ep_size=ep_size(), ep_rank=ep_rank())


def executor_arguments(config: EngineConfig) -> tuple[str, dict]:
    """Return the selected backend and constructor args derived from EngineConfig."""
    backend = get_executor_backend(config.execution.mode)
    backend.validate(config)
    return config.execution.mode, backend.arguments(config)


def build_executor(config: EngineConfig):
    mode, kwargs = executor_arguments(config)
    from auto_infer.distributed.parallel_state import init_distributed
    init_distributed(config.parallel)
    return get_executor_backend(mode).load()(**kwargs)
