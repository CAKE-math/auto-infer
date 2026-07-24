import torch

from auto_infer.harness.capabilities import match_capabilities
from auto_infer.harness.inspect import inspect_model
from auto_infer.harness.package import generate_package
from auto_infer.models.base import BaseCausalLM
from tests.test_harness_inspect import (
    _gqa_config,
    _gqa_keys,
    _write_model,
)


def test_logits_partition_exposes_replicated_vocabulary_range():
    model = BaseCausalLM()
    model.w = {"lm_head.weight": torch.randn(7, 3)}
    hidden = torch.randn(2, 3)

    logits, vocabulary = model.logits_partition(hidden)

    torch.testing.assert_close(logits, model.logits(hidden))
    assert vocabulary == (0, 7)


def test_generated_gqa_package_declares_bf16_tensor_parallel_support(
    tmp_path,
):
    model = _write_model(tmp_path, _gqa_config(), _gqa_keys())
    manifest = inspect_model(model)
    manifest["capability"] = match_capabilities(manifest)

    package = generate_package(manifest, tmp_path / "package")

    assert package["execution"]["parallel"] == {
        "tensor": {
            "status": "supported",
            "dtype": "bfloat16",
            "max_size": 8,
            "modes": ["recompute", "paged"],
        },
        "expert": {"status": "unsupported"},
    }


def test_generated_mla_package_declares_tp_unsupported_ep_supported(
    tmp_path,
):
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
    manifest["capability"] = match_capabilities(manifest)

    package = generate_package(manifest, tmp_path / "package")

    assert package["execution"]["parallel"] == {
        "tensor": {"status": "unsupported"},
        "expert": {"status": "supported", "dtype": "bfloat16"},
    }
