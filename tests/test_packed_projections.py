import torch
from types import SimpleNamespace

from auto_infer.models.qwen2 import Qwen2Model, pack_qwen2_projections
from auto_infer.layers.attention.gqa import _split_qkv


def _weights(*, bias=True):
    torch.manual_seed(19)
    p = "model.layers.0."
    weights = {
        p + "self_attn.q_proj.weight": torch.randn(8, 6),
        p + "self_attn.k_proj.weight": torch.randn(4, 6),
        p + "self_attn.v_proj.weight": torch.randn(4, 6),
        p + "mlp.gate_proj.weight": torch.randn(10, 6),
        p + "mlp.up_proj.weight": torch.randn(10, 6),
    }
    if bias:
        weights.update({
            p + "self_attn.q_proj.bias": torch.randn(8),
            p + "self_attn.k_proj.bias": torch.randn(4),
            p + "self_attn.v_proj.bias": torch.randn(4),
        })
    return weights


def test_pack_qkv_and_gate_up_preserves_linear_results():
    weights = _weights()
    original = dict(weights)
    x = torch.randn(3, 6)

    pack_qwen2_projections(weights, num_layers=1)

    p = "model.layers.0."
    qkv = x @ weights[p + "self_attn.qkv_proj.weight"].t()
    qkv += weights[p + "self_attn.qkv_proj.bias"]
    expected_qkv = torch.cat([
        x @ original[p + f"self_attn.{name}_proj.weight"].t()
        + original[p + f"self_attn.{name}_proj.bias"]
        for name in ("q", "k", "v")
    ], dim=-1)
    torch.testing.assert_close(qkv, expected_qkv)

    gate_up = x @ weights[p + "mlp.gate_up_proj.weight"].t()
    expected_gate_up = torch.cat([
        x @ original[p + "mlp.gate_proj.weight"].t(),
        x @ original[p + "mlp.up_proj.weight"].t(),
    ], dim=-1)
    torch.testing.assert_close(gate_up, expected_gate_up)


def test_pack_removes_source_weights_and_supports_biasless_qwen3():
    weights = _weights(bias=False)

    pack_qwen2_projections(weights, num_layers=1)

    assert "model.layers.0.self_attn.qkv_proj.bias" not in weights
    for name in ("q_proj", "k_proj", "v_proj"):
        assert f"model.layers.0.self_attn.{name}.weight" not in weights
    for name in ("gate_proj", "up_proj"):
        assert f"model.layers.0.mlp.{name}.weight" not in weights


def test_pack_is_idempotent():
    weights = _weights()
    pack_qwen2_projections(weights, num_layers=1)
    first_qkv = weights["model.layers.0.self_attn.qkv_proj.weight"]

    pack_qwen2_projections(weights, num_layers=1)

    assert weights["model.layers.0.self_attn.qkv_proj.weight"] is first_qkv


def test_pack_quantized_weights_concatenates_output_dimension_and_scales():
    weights = _weights(bias=False)
    expected = {}
    for name, value in list(weights.items()):
        scale = value.abs().amax(dim=1).float()
        quantized = (value.round().to(torch.int8).t().contiguous(), scale)
        weights[name] = quantized
        expected[name] = quantized

    pack_qwen2_projections(weights, num_layers=1)

    p = "model.layers.0."
    packed_w, packed_s = weights[p + "self_attn.qkv_proj.weight"]
    assert packed_w.shape == (6, 16)
    torch.testing.assert_close(packed_s, torch.cat([
        expected[p + f"self_attn.{name}_proj.weight"][1]
        for name in ("q", "k", "v")
    ]))
    gate_w, gate_s = weights[p + "mlp.gate_up_proj.weight"]
    assert gate_w.shape == (6, 20)
    assert gate_s.shape == (20,)


def test_split_qkv_returns_views_with_expected_shapes():
    projected = torch.randn(2, 16, dtype=torch.bfloat16)
    q, k, v = _split_qkv(projected, 8, 4)

    assert (q.shape, k.shape, v.shape) == ((2, 8), (2, 4), (2, 4))
    assert q.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()
    assert k.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()
    assert v.untyped_storage().data_ptr() == projected.untyped_storage().data_ptr()


def test_tp_sharding_accepts_biasless_qwen3_attention():
    weights = _weights(bias=False)
    p = "model.layers.0."
    weights[p + "self_attn.o_proj.weight"] = torch.randn(6, 8)
    weights[p + "mlp.down_proj.weight"] = torch.randn(6, 10)
    cfg = SimpleNamespace(
        num_layers=1, head_dim=4, num_heads=2,
        num_kv_heads=1, intermediate_size=10)

    sharded = Qwen2Model._shard_tp(weights, cfg, r=0, tp=1)

    assert sharded[p + "self_attn.q_proj.weight"].shape == (8, 6)
    assert p + "self_attn.q_proj.bias" not in sharded
