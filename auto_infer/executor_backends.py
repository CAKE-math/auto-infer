"""Neutral execution-backend specifications shared by config and engine."""
from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class ExecutorBackend:
    validate: Callable[[object], None]
    arguments: Callable[[object], dict]
    load: Callable[[], type]


_BACKENDS: dict[str, ExecutorBackend] = {}


def register_executor_backend(name: str, backend: ExecutorBackend) -> None:
    if not name:
        raise ValueError("executor backend name must not be empty")
    if name in _BACKENDS:
        raise ValueError(f"executor backend already registered: {name}")
    _BACKENDS[name] = backend


def get_executor_backend(name: str) -> ExecutorBackend:
    try:
        return _BACKENDS[name]
    except KeyError:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"unsupported execution mode: {name}; registered modes: {choices}"
        ) from None


def has_executor_backend(name: str) -> bool:
    return name in _BACKENDS


def _load(module: str, name: str):
    return lambda: getattr(import_module(module), name)


def _common(config) -> dict:
    return {
        "model_path": config.model.model_path,
        "device_index": config.execution.device_index,
        "dtype": config.model.dtype,
    }


def _cached(config) -> dict:
    return {
        **_common(config),
        "num_blocks": config.cache.num_blocks,
        "block_size": config.cache.block_size,
    }


def _paged(config) -> dict:
    arguments = {
        **_cached(config),
        "max_num_batched_tokens": config.scheduler.max_num_batched_tokens,
        "max_num_seqs": config.scheduler.max_num_seqs,
        "max_model_len": config.model.max_model_len,
    }
    if config.spec_decode is not None:
        arguments["num_speculative_tokens"] = (
            config.spec_decode.num_speculative_tokens)
    return arguments


def _graph(config) -> dict:
    return {
        **_cached(config),
        "max_gear": config.execution.max_gear,
        "max_prefill_tokens": config.execution.max_prefill_tokens,
        "max_model_len": config.model.max_model_len,
        "force_eager": config.execution.force_eager,
        "async_slots": (
            config.async_batches if config.async_scheduling else 1),
        "max_num_seqs": config.scheduler.max_num_seqs,
    }


def _graph_mtp(config) -> dict:
    return {
        **_cached(config),
        "max_gear": config.execution.max_gear,
        "max_model_len": config.model.max_model_len,
        "num_speculative_tokens": config.spec_decode.num_speculative_tokens,
    }


def _validate_recompute(config) -> None:
    if config.spec_decode is not None:
        raise ValueError("spec_decode requires paged or graph_mtp execution")


def _validate_paged(config) -> None:
    return None


def _validate_graph(config) -> None:
    if config.spec_decode is not None:
        raise ValueError("spec_decode with graph execution requires graph_mtp mode")


def _validate_graph_mtp(config) -> None:
    if config.spec_decode is None:
        raise ValueError("graph_mtp execution requires spec_decode")
    if config.execution.force_eager:
        raise ValueError("graph_mtp execution does not support force_eager")
    from auto_infer.spec_decode.geometry import validate_graph_mtp_depth
    validate_graph_mtp_depth(config.spec_decode.num_speculative_tokens)


register_executor_backend(
    "recompute",
    ExecutorBackend(
        _validate_recompute,
        _common,
        _load("auto_infer.engine.npu_executor", "NpuExecutor"),
    ),
)
register_executor_backend(
    "paged",
    ExecutorBackend(
        _validate_paged,
        _paged,
        _load("auto_infer.worker.model_runner", "PagedNpuExecutor"),
    ),
)
register_executor_backend(
    "graph",
    ExecutorBackend(
        _validate_graph,
        _graph,
        _load("auto_infer.worker.graph_decode_runner", "GraphPagedNpuExecutor"),
    ),
)
register_executor_backend(
    "graph_mtp",
    ExecutorBackend(
        _validate_graph_mtp,
        _graph_mtp,
        _load("auto_infer.worker.graph_mtp_runner", "GraphMtpPagedNpuExecutor"),
    ),
)
