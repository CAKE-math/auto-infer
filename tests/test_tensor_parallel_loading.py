import os
import tempfile
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from auto_infer.models.loader import load_sharded
from auto_infer.models.parallel import TensorParallelPlan
from auto_infer.models.qwen2 import Qwen2Model


def _config(**overrides):
    values = {
        "num_heads": 8,
        "num_kv_heads": 4,
        "intermediate_size": 32,
        "head_dim": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_tp_plan_rejects_invalid_rank():
    with pytest.raises(ValueError, match="rank"):
        TensorParallelPlan.for_qwen(_config(), rank=2, size=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_heads", 7),
        ("num_kv_heads", 3),
        ("intermediate_size", 31),
    ],
)
def test_qwen_tp_plan_rejects_non_divisible_dimensions(field, value):
    with pytest.raises(ValueError, match=field):
        TensorParallelPlan.for_qwen(
            _config(**{field: value}), rank=0, size=2)


def test_qwen_tp_plan_maps_column_and_row_parallel_weights():
    plan = TensorParallelPlan.for_qwen(_config(), rank=1, size=2)

    assert plan.slice_spec(
        "model.layers.0.self_attn.q_proj.weight") == (0, 32, 32)
    assert plan.slice_spec(
        "model.layers.0.self_attn.k_proj.bias") == (0, 16, 16)
    assert plan.slice_spec(
        "model.layers.0.self_attn.o_proj.weight") == (1, 32, 32)
    assert plan.slice_spec(
        "model.layers.0.mlp.gate_proj.weight") == (0, 16, 16)
    assert plan.slice_spec(
        "model.layers.0.mlp.down_proj.weight") == (1, 16, 16)


def test_qwen_tp_plan_leaves_replicated_and_mtp_weights_untouched():
    plan = TensorParallelPlan.for_qwen(_config(), rank=0, size=2)

    assert plan.slice_spec("model.embed_tokens.weight") is None
    assert plan.slice_spec("lm_head.weight") is None
    assert plan.slice_spec(
        "model.mtp_layers.0.self_attn.q_proj.weight") is None


def test_load_sharded_reads_only_the_requested_tensor_partition():
    config = _config(
        num_heads=4, num_kv_heads=2, intermediate_size=8, head_dim=2)
    tensors = {
        "model.layers.0.self_attn.q_proj.weight":
            torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "model.layers.0.self_attn.o_proj.weight":
            torch.arange(32, dtype=torch.float32).reshape(4, 8),
        "model.layers.0.mlp.gate_proj.weight":
            torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "model.embed_tokens.weight":
            torch.arange(20, dtype=torch.float32).reshape(5, 4),
    }
    with tempfile.TemporaryDirectory() as directory:
        save_file(tensors, os.path.join(directory, "model.safetensors"))
        full = load_sharded(
            directory, lambda _: True, device="cpu",
            dtype=torch.float32, max_workers=1)
        rank_shards = []
        for rank in range(2):
            plan = TensorParallelPlan.for_qwen(config, rank=rank, size=2)
            rank_shards.append(load_sharded(
                directory, lambda _: True, device="cpu",
                dtype=torch.float32, max_workers=1,
                slicer=plan.slice_spec))

    assert torch.equal(
        torch.cat([
            shard["model.layers.0.self_attn.q_proj.weight"]
            for shard in rank_shards
        ], dim=0),
        full["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        torch.cat([
            shard["model.layers.0.self_attn.o_proj.weight"]
            for shard in rank_shards
        ], dim=1),
        full["model.layers.0.self_attn.o_proj.weight"],
    )
    assert torch.equal(
        torch.cat([
            shard["model.layers.0.mlp.gate_proj.weight"]
            for shard in rank_shards
        ], dim=0),
        full["model.layers.0.mlp.gate_proj.weight"],
    )
    for shard in rank_shards:
        assert torch.equal(
            shard["model.embed_tokens.weight"],
            full["model.embed_tokens.weight"],
        )


def _write_tp_qwen(directory):
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 16,
        "vocab_size": 12,
        "architectures": ["Qwen2ForCausalLM"],
        "tie_word_embeddings": True,
    }
    with open(os.path.join(directory, "config.json"), "w") as stream:
        json.dump(config, stream)
    weights = {
        "model.embed_tokens.weight": torch.randn(12, 8),
        "model.norm.weight": torch.randn(8),
        "model.layers.0.input_layernorm.weight": torch.randn(8),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(8),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(4),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(4),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 16),
    }
    save_file(weights, os.path.join(directory, "model.safetensors"))


def test_qwen_loads_rank_shards_before_projection_packing(monkeypatch):
    import auto_infer.models.loader as loader

    real_load = loader.load_sharded
    observed = []

    def recording_load(*args, **kwargs):
        observed.append(kwargs.get("slicer"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(loader, "load_sharded", recording_load)
    with tempfile.TemporaryDirectory() as directory:
        _write_tp_qwen(directory)
        model = Qwen2Model.from_pretrained(
            directory, torch.device("cpu"), torch.float32,
            tp_rank=1, tp_size=2)

    assert observed and observed[0] is not None
    assert model.w[
        "model.layers.0.self_attn.qkv_proj.weight"].shape == (8, 8)
    assert model.w[
        "model.layers.0.self_attn.o_proj.weight"].shape == (8, 4)
    assert model.w[
        "model.layers.0.mlp.gate_up_proj.weight"].shape == (16, 8)
    assert model.w[
        "model.layers.0.mlp.down_proj.weight"].shape == (8, 8)


def _model_directory(directory, architecture="UnsupportedForCausalLM"):
    with open(os.path.join(directory, "config.json"), "w") as stream:
        json.dump({"architectures": [architecture]}, stream)


def test_factory_rejects_model_without_tensor_parallel_capability(monkeypatch):
    from auto_infer.distributed import parallel_state
    from auto_infer.engine.factory import load_model
    import auto_infer.models.registry as registry
    import auto_infer.platform as platform

    class Unsupported:
        SUPPORTS_TENSOR_PARALLEL = False

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("unsupported loader must not run")

    monkeypatch.setattr(registry, "get_model_class", lambda _: Unsupported)
    monkeypatch.setattr(platform, "npu_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(platform, "default_dtype", lambda _: torch.bfloat16)
    monkeypatch.setattr(parallel_state, "tp_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "tp_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "ep_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "ep_size", lambda: 1)
    with tempfile.TemporaryDirectory() as directory:
        _model_directory(directory)
        with pytest.raises(ValueError, match="tensor parallel"):
            load_model(directory, 0, "bfloat16")


def test_factory_rejects_model_without_capability_attribute(monkeypatch):
    from auto_infer.distributed import parallel_state
    from auto_infer.engine.factory import load_model
    import auto_infer.models.registry as registry
    import auto_infer.platform as platform

    class LegacyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("unsupported loader must not run")

    monkeypatch.setattr(registry, "get_model_class", lambda _: LegacyModel)
    monkeypatch.setattr(platform, "npu_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(platform, "default_dtype", lambda _: torch.bfloat16)
    monkeypatch.setattr(parallel_state, "tp_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "tp_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "ep_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "ep_size", lambda: 1)
    with tempfile.TemporaryDirectory() as directory:
        _model_directory(directory)
        with pytest.raises(ValueError, match="tensor parallel"):
            load_model(directory, 0, "bfloat16")


def test_qwen_rejects_zero_tp_size_before_dividing():
    with tempfile.TemporaryDirectory() as directory:
        _write_tp_qwen(directory)
        with pytest.raises(ValueError, match="size"):
            Qwen2Model.from_pretrained(
                directory, torch.device("cpu"), torch.float32,
                tp_rank=0, tp_size=0)


def test_factory_passes_tp_and_ep_coordinates_to_supported_model(monkeypatch):
    from auto_infer.distributed import parallel_state
    from auto_infer.engine.factory import load_model
    import auto_infer.models.registry as registry
    import auto_infer.platform as platform

    class Supported:
        SUPPORTS_TENSOR_PARALLEL = True

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return kwargs

    monkeypatch.setattr(registry, "get_model_class", lambda _: Supported)
    monkeypatch.setattr(platform, "npu_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(platform, "default_dtype", lambda _: torch.bfloat16)
    monkeypatch.setattr(parallel_state, "tp_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "tp_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "ep_rank", lambda: 3)
    monkeypatch.setattr(parallel_state, "ep_size", lambda: 4)
    with tempfile.TemporaryDirectory() as directory:
        _model_directory(directory)
        kwargs = load_model(directory, 0, "bfloat16")

    assert kwargs["tp_rank"] == 1
    assert kwargs["tp_size"] == 2
    assert kwargs["ep_rank"] == 3
    assert kwargs["ep_size"] == 4
