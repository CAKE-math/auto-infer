import json

import pytest

from auto_infer.config import (
    EngineConfig, ExecutionConfig, ModelConfig, ParallelConfig,
    SpecDecodeConfig)
from auto_infer.errors import ConfigurationError
from auto_infer.executor_backends import ExecutorBackend, register_executor_backend
from auto_infer.engine.factory import (
    build_executor,
    executor_arguments,
    load_model,
)
from auto_infer.engine.engine_core import EngineCore


def test_executor_arguments_are_derived_from_one_engine_config():
    cfg = EngineConfig(
        model=ModelConfig("/models/qwen", max_model_len=1234, dtype="float16"),
        execution=ExecutionConfig(mode="paged", device_index=3, max_gear=8),
    )
    cls_name, kwargs = executor_arguments(cfg)
    assert cls_name == "paged"
    assert kwargs == {
        "model_path": "/models/qwen",
        "num_blocks": cfg.cache.num_blocks,
        "block_size": cfg.cache.block_size,
        "device_index": 3,
        "dtype": "float16",
        "max_num_batched_tokens": cfg.scheduler.max_num_batched_tokens,
        "max_num_seqs": cfg.scheduler.max_num_seqs,
        "max_model_len": 1234,
    }


def test_graph_mtp_requires_spec_decode_and_graph_rejects_spec_decode():
    base = ModelConfig("/models/m")
    with pytest.raises(ValueError, match="requires spec_decode"):
        executor_arguments(EngineConfig(
            model=base, execution=ExecutionConfig(mode="graph_mtp")))
    with pytest.raises(ValueError, match="graph_mtp"):
        executor_arguments(EngineConfig(
            model=base, execution=ExecutionConfig(mode="graph"),
            spec_decode=SpecDecodeConfig()))
    with pytest.raises(ValueError, match="paged or graph_mtp"):
        executor_arguments(EngineConfig(
            model=base, execution=ExecutionConfig(mode="recompute"),
            spec_decode=SpecDecodeConfig()))
    with pytest.raises(ValueError, match="does not support force_eager"):
        executor_arguments(EngineConfig(
            model=base,
            execution=ExecutionConfig(mode="graph_mtp", force_eager=True),
            spec_decode=SpecDecodeConfig()))


def test_graph_mtp_factory_propagates_speculative_depth():
    cfg = EngineConfig(
        model=ModelConfig("/models/m"),
        execution=ExecutionConfig(mode="graph_mtp"),
        spec_decode=SpecDecodeConfig(num_speculative_tokens=2))

    _, kwargs = executor_arguments(cfg)

    assert kwargs["num_speculative_tokens"] == 2


def test_graph_executor_propagates_independent_graph_limits():
    cfg = EngineConfig(
        model=ModelConfig("/models/qwen"),
        execution=ExecutionConfig(
            mode="graph", max_gear=32, max_prefill_tokens=256),
    )

    _, kwargs = executor_arguments(cfg)

    assert kwargs["max_gear"] == 32
    assert kwargs["max_prefill_tokens"] == 256


def test_graph_mtp_rejects_depth_beyond_npu_verified_boundary():
    config = EngineConfig(
        model=ModelConfig("/models/m"),
        execution=ExecutionConfig(mode="graph_mtp"),
        spec_decode=SpecDecodeConfig(num_speculative_tokens=3))

    with pytest.raises(ValueError, match="verified maximum is 2"):
        executor_arguments(config)


def test_models_do_not_own_runtime_backend_factories():
    from auto_infer.models.base import BaseCausalLM
    for name in ("make_dense_backend", "make_attention_backend", "make_graph_backend"):
        assert not hasattr(BaseCausalLM, name)


def test_registered_executor_needs_no_factory_branch():
    class SyntheticExecutor:
        def __init__(self, marker):
            self.marker = marker

    register_executor_backend(
        "synthetic-test",
        ExecutorBackend(
            validate=lambda config: None,
            arguments=lambda config: {"marker": config.execution.max_gear},
            load=lambda: SyntheticExecutor,
        ),
    )
    cfg = EngineConfig(
        model=ModelConfig("/models/synthetic"),
        execution=ExecutionConfig(mode="synthetic-test", max_gear=7),
    )

    mode, kwargs = executor_arguments(cfg)
    executor = build_executor(cfg)

    assert (mode, kwargs) == ("synthetic-test", {"marker": 7})
    assert isinstance(executor, SyntheticExecutor)
    assert executor.marker == 7


def test_executor_registry_rejects_duplicate_names():
    backend = ExecutorBackend(
        validate=lambda config: None,
        arguments=lambda config: {},
        load=lambda: object,
    )
    register_executor_backend("duplicate-executor-test", backend)
    with pytest.raises(ValueError, match="already registered"):
        register_executor_backend("duplicate-executor-test", backend)


def test_build_executor_initializes_parallel_before_model_construction(
        monkeypatch):
    events = []

    class DistributedExecutor:
        def __init__(self):
            events.append("executor")

    register_executor_backend(
        "distributed-test",
        ExecutorBackend(
            validate=lambda config: None,
            arguments=lambda config: {},
            load=lambda: DistributedExecutor,
        ),
    )
    cfg = EngineConfig(
        model=ModelConfig("/models/distributed"),
        execution=ExecutionConfig(mode="distributed-test"),
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.init_distributed",
        lambda parallel: events.append(("distributed", parallel)))

    build_executor(cfg)

    assert events == [("distributed", cfg.parallel), "executor"]


def test_build_executor_registers_model_package_before_distributed(
        monkeypatch):
    events = []

    class PackageExecutor:
        pass

    register_executor_backend(
        "package-order-test",
        ExecutorBackend(
            validate=lambda config: None,
            arguments=lambda config: {},
            load=lambda: PackageExecutor,
        ),
    )
    cfg = EngineConfig(
        model=ModelConfig(
            "/models/package",
            model_package="/packages/example",
        ),
        execution=ExecutionConfig(mode="package-order-test"),
    )
    monkeypatch.setattr(
        "auto_infer.models.registry.register_package",
        lambda package, model: events.append(("package", package, model)),
    )
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.init_distributed",
        lambda parallel: events.append("distributed"),
    )

    build_executor(cfg)

    assert events == [
        ("package", "/packages/example", "/models/package"),
        "distributed",
    ]


def test_engine_core_has_no_distributed_bootstrap_side_effect(monkeypatch):
    calls = []
    config = EngineConfig(model=ModelConfig("/models/injected"))
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.init_distributed",
        lambda parallel: calls.append(parallel))

    EngineCore(config, object())

    assert calls == []


def test_build_executor_rejects_parallel_world_mismatch(monkeypatch):
    cfg = EngineConfig(
        model=ModelConfig("/models/qwen"),
        parallel=ParallelConfig(ep_size=2),
        execution=ExecutionConfig(mode="paged"),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(ConfigurationError, match="WORLD_SIZE=1"):
        build_executor(cfg)


def test_load_model_passes_initialized_ep_shard_to_model(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["SyntheticForCausalLM"]}))
    received = {}

    class SyntheticModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            received.update(path=path, **kwargs)
            return cls()

    monkeypatch.setattr(
        "auto_infer.models.registry.get_model_class",
        lambda architecture: SyntheticModel)
    monkeypatch.setattr(
        "auto_infer.platform.npu_device", lambda index: f"npu:{index}")
    monkeypatch.setattr(
        "auto_infer.platform.default_dtype", lambda dtype: dtype)
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.ep_size", lambda: 4)
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.ep_rank", lambda: 2)

    load_model(str(tmp_path), 3, "bfloat16")

    assert received == {
        "path": str(tmp_path), "device": "npu:3", "dtype": "bfloat16",
        "ep_size": 4, "ep_rank": 2,
    }
