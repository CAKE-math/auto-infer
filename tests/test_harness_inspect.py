import json

import pytest

from auto_infer.harness.capabilities import match_capabilities
from auto_infer.harness.inspect import inspect_model


def _write_model(tmp_path, config, keys=()):
    model = tmp_path / config["model_type"]
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config))
    if keys:
        (model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model-00001-of-00001.safetensors"
                           for key in keys}
        }))
    return model


def _gqa_keys(*, qk_norm=False):
    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    }
    if qk_norm:
        keys |= {
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
        }
    return keys


def _gqa_config(**extra):
    config = {
        "model_type": "example",
        "architectures": ["ExampleForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
    }
    config.update(extra)
    return config


def test_inspect_and_match_qwen2_compatible_gqa(tmp_path):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())

    manifest = inspect_model(model)
    match = match_capabilities(manifest)

    assert manifest["architecture"] == {
        "attention": "gqa",
        "head_dim": 64,
        "hidden_size": 1024,
        "num_heads": 16,
        "num_kv_heads": 4,
        "num_layers": 24,
        "position_embedding": "rope",
        "type": "decoder_only",
    }
    assert manifest["features"] == {
        "moe": False,
        "mtp": False,
        "sliding_window": False,
    }
    assert manifest["cache"]["type"] == "paged_kv"
    assert manifest["weights"]["evidence"] == "index"
    assert match["status"] == "supported"
    assert match["template"] == "gqa-swiglu-v1"
    assert match["entrypoint"] == "auto_infer.models.qwen2:Qwen2Model"
    assert match["missing"] == []


def test_inspect_selects_qwen3_contract_from_head_dim_and_qk_norm(tmp_path):
    config = _gqa_config(
        model_type="example_qwen3",
        architectures=["ExampleQwen3ForCausalLM"],
        head_dim=128,
    )
    model = _write_model(tmp_path, config, _gqa_keys(qk_norm=True))

    match = match_capabilities(inspect_model(model))

    assert match["status"] == "supported"
    assert match["template"] == "gqa-qknorm-v1"
    assert match["entrypoint"] == "auto_infer.models.qwen3:Qwen3Model"


def test_inspect_and_match_deepseek_style_mla_moe(tmp_path):
    config = {
        "model_type": "example_mla",
        "architectures": ["ExampleMlaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "kv_lora_rank": 512,
        "q_lora_rank": 256,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 100000,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "first_k_dense_replace": 1,
    }
    keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    model = _write_model(tmp_path, config, keys)

    manifest = inspect_model(model)
    match = match_capabilities(manifest)

    assert manifest["architecture"]["attention"] == "mla"
    assert manifest["features"]["moe"] is True
    assert match["status"] == "supported"
    assert match["template"] == "mla-moe-v1"
    assert match["entrypoint"] == (
        "auto_infer.models.deepseek_v2:DeepseekV2Model"
    )


@pytest.mark.parametrize(
    "mutation,missing",
    [
        ({"num_key_value_heads": 3}, "attention.head_divisibility"),
        ({"num_attention_heads": None}, "config.num_attention_heads"),
    ],
)
def test_capability_match_is_partial_when_contract_cannot_be_proven(
    tmp_path, mutation, missing
):
    config = _gqa_config(**mutation)
    model = _write_model(tmp_path, config, _gqa_keys())

    match = match_capabilities(inspect_model(model))

    assert match["status"] == "partial"
    assert missing in match["missing"]
    assert match["entrypoint"] is None


def test_capability_match_requires_weight_layout_evidence(tmp_path):
    model = _write_model(tmp_path, _gqa_config())

    manifest = inspect_model(model)
    match = match_capabilities(manifest)

    assert manifest["weights"]["evidence"] == "absent"
    assert match["status"] == "partial"
    assert "weights.standard_layout" in match["missing"]
