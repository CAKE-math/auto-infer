"""Host unit test for Qwen2Model wiring that does NOT need torch_npu: the
model-owned `layer_prefix` convention + backend sharing `model.w`.

`forward(ctx)` itself is NPU-only now (fused npu rms_norm/swiglu, no CPU
fallback), so the old CPU hand-reference forward-parity tests were removed —
forward correctness is covered end-to-end by scripts/smoke_qwen2.py /
smoke_qwen3.py HF next-token parity on npu2.
"""
import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from auto_infer.models.qwen2 import Qwen2Model, pack_qwen2_projections
from auto_infer.layers.mlp import _gate_up_projection

HIDDEN, HEADS, KV_HEADS, INTER, VOCAB, LAYERS = 8, 2, 1, 16, 12, 2
HEAD_DIM = HIDDEN // HEADS
EPS = 1e-6


def _write_tiny_qwen2(d):
    cfg = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "intermediate_size": INTER, "vocab_size": VOCAB,
        "architectures": ["Qwen2ForCausalLM"], "rms_norm_eps": EPS,
        "rope_theta": 1000000.0, "tie_word_embeddings": True,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(42)
    w = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
         "model.norm.weight": torch.randn(HIDDEN) * 0.1 + 1.0}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.randn(HIDDEN) * 0.1 + 1.0
        w[p + "post_attention_layernorm.weight"] = torch.randn(HIDDEN) * 0.1 + 1.0
        w[p + "self_attn.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN) * 0.1
        w[p + "self_attn.q_proj.bias"] = torch.randn(HEADS * HEAD_DIM) * 0.1
        w[p + "self_attn.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN) * 0.1
        w[p + "self_attn.k_proj.bias"] = torch.randn(KV_HEADS * HEAD_DIM) * 0.1
        w[p + "self_attn.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN) * 0.1
        w[p + "self_attn.v_proj.bias"] = torch.randn(KV_HEADS * HEAD_DIM) * 0.1
        w[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, HEADS * HEAD_DIM) * 0.1
        w[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
        w[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
        w[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER) * 0.1
    save_file(w, os.path.join(d, "model.safetensors"))
    return w


def test_qwen2_layer_prefix_convention():
    """The model owns its per-layer weight-name convention (fed to the attention
    backends); there is no per-layer params struct — forward derives the prefix
    from the layer index and the backends index model.w by name."""
    with tempfile.TemporaryDirectory() as d:
        _write_tiny_qwen2(d)
        model = Qwen2Model.from_pretrained(d, device=torch.device("cpu"), dtype=torch.float32)
        assert model.layer_prefix(0) == "model.layers.0."
        assert model.layer_prefix(3) == "model.layers.3."
        from auto_infer.layers.attention.registry import build_attention_backend
        be, _ = build_attention_backend(model, "paged", num_blocks=1, block_size=16)
        assert be.w is model.w                          # backend shares the model's dict
        assert be.layer_prefix(2) == "model.layers.2."


def test_projection_packing_includes_mtp_layers():
    w = {}
    prefix = "model.mtp_layers.0."
    attn = prefix + "self_attn."
    mlp = prefix + "mlp."
    for name, rows in (("q", 8), ("k", 4), ("v", 4)):
        w[attn + f"{name}_proj.weight"] = torch.randn(rows, HIDDEN)
        w[attn + f"{name}_proj.bias"] = torch.randn(rows)
    w[mlp + "gate_proj.weight"] = torch.randn(INTER, HIDDEN)
    w[mlp + "up_proj.weight"] = torch.randn(INTER, HIDDEN)

    pack_qwen2_projections(w, num_layers=0)

    assert w[attn + "qkv_proj.weight"].shape == (16, HIDDEN)
    assert w[attn + "qkv_proj.bias"].shape == (16,)
    assert w[mlp + "gate_up_proj.weight"].shape == (2 * INTER, HIDDEN)
    assert attn + "q_proj.weight" not in w
    assert mlp + "gate_proj.weight" not in w

    x = torch.randn(3, HIDDEN)
    expected = torch.cat([
        x @ w[mlp + "gate_up_proj.weight"][:INTER].t(),
        x @ w[mlp + "gate_up_proj.weight"][INTER:].t(),
    ], dim=-1)
    assert torch.allclose(_gate_up_projection(x, w, mlp), expected)
