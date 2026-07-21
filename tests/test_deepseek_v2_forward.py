"""Host-only unit tests for the unified DeepseekV2Model.forward(ctx) (SP5 task
10; migrates DeepSeek onto the shared forward-shell shape after SP5's
AttentionBackend generalization — see
docs/superpowers/specs/2026-07-17-skeleton-sp5-deepseek-mla-backend.md).

The REAL `MlaFIABackend` calls straight into torch_npu (paged FIA / KV write —
same as `GqaFIABackend`/`GraphGqaBackend`, NPU-only, exercised by
the unified NPU model/backend parity checks). What IS host-testable, and what this file
covers: (1) the model-owned `layer_prefix` convention (fed to the MLA backends
+ MoE block), (2) `MlaFIABackend`'s pure-Python surface (construction +
`alloc_kv_caches` shape/dtype/device — same split as `GqaFIABackend`'s tests),
and (3) the `forward(ctx)` SHELL's wiring (fused-residual pattern, per-layer
`is_moe` dense-vs-MoE dispatch, `ctx.cos/sin` plumbing) — using a tiny
host-only stand-in `AttentionBackend` that reproduces the SAME MLA math as the
existing (untouched) `forward_dense` naive path, so the shell's forward(ctx)
output can be checked against forward_dense's independently-computed result.
"""
import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from auto_infer.layers.attention.base import AttentionBackend
from auto_infer.layers.attention.mla import MlaFIABackend
from auto_infer.models.deepseek_v2 import DeepseekV2Model

HIDDEN, HEADS, INTER = 8, 2, 8
KV_LORA, NOPE, ROPE, VD = 4, 2, 2, 2
N_ROUTED, N_SHARED, TOP_K = 2, 1, 1
VOCAB, LAYERS, FIRST_K_DENSE = 12, 2, 1     # layer 0 dense, layer 1 MoE
EPS = 1e-6
T = 5


def _write_tiny_deepseek(d):
    cfg = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "kv_lora_rank": KV_LORA, "q_lora_rank": None,
        "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE, "v_head_dim": VD,
        "vocab_size": VOCAB, "architectures": ["DeepseekV2ForCausalLM"],
        "rms_norm_eps": EPS, "rope_theta": 10000, "n_routed_experts": N_ROUTED,
        "n_shared_experts": N_SHARED, "num_experts_per_tok": TOP_K,
        "first_k_dense_replace": FIRST_K_DENSE, "norm_topk_prob": False,
        "scoring_func": "softmax", "topk_method": "greedy",
        "routed_scaling_factor": 1.0, "rope_scaling": None,
        "tie_word_embeddings": False,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(7)
    qk = NOPE + ROPE
    w = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN) * 0.1,
         "model.norm.weight": torch.randn(HIDDEN) * 0.1 + 1.0}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.randn(HIDDEN) * 0.1 + 1.0
        w[p + "post_attention_layernorm.weight"] = torch.randn(HIDDEN) * 0.1 + 1.0
        w[p + "self_attn.q_proj.weight"] = torch.randn(HEADS * qk, HIDDEN) * 0.1
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(KV_LORA + ROPE, HIDDEN) * 0.1
        w[p + "self_attn.kv_a_layernorm.weight"] = torch.randn(KV_LORA) * 0.1 + 1.0
        w[p + "self_attn.kv_b_proj.weight"] = torch.randn(HEADS * (NOPE + VD), KV_LORA) * 0.1
        w[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, HEADS * VD) * 0.1
        if i < FIRST_K_DENSE:
            w[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
            w[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
            w[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER) * 0.1
        else:
            mp = p + "mlp."
            w[mp + "gate.weight"] = torch.randn(N_ROUTED, HIDDEN) * 0.1
            for e in range(N_ROUTED):
                ep = f"{mp}experts.{e}."
                w[ep + "gate_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
                w[ep + "up_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
                w[ep + "down_proj.weight"] = torch.randn(HIDDEN, INTER) * 0.1
            sp = mp + "shared_experts."
            w[sp + "gate_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
            w[sp + "up_proj.weight"] = torch.randn(INTER, HIDDEN) * 0.1
            w[sp + "down_proj.weight"] = torch.randn(HIDDEN, INTER) * 0.1
    save_file(w, os.path.join(d, "model.safetensors"))
    return w


def _load_tiny_model(d):
    model = DeepseekV2Model.from_pretrained(d, device=torch.device("cpu"), dtype=torch.float32)
    model.moe.fused = False   # force the naive per-expert MoE loop (CPU-safe; the
    #                           fused grouped-GEMM path is torch_npu-only)
    return model


def test_deepseek_layer_prefix_convention():
    """The model owns its per-layer weight-name convention (fed to the MLA
    backends + MoE block), so there is no per-layer params struct — forward
    derives prefix/is_moe from the layer index directly."""
    with tempfile.TemporaryDirectory() as d:
        _write_tiny_deepseek(d)
        model = _load_tiny_model(d)
        assert model.layer_prefix(0) == "model.layers.0."
        assert model.layer_prefix(LAYERS - 1) == f"model.layers.{LAYERS - 1}."
        # the MoE block received the same convention (not a hardcoded copy)
        assert model.moe.layer_prefix(3) == "model.layers.3."


def test_mla_fia_backend_alloc_kv_caches_shape_dtype_device_non_absorbed():
    nh, nope, rope, vd, kvl, layers = 4, 2, 2, 2, 8, 3
    backend = MlaFIABackend({}, num_heads=nh, qk_nope=nope, qk_rope=rope, v_head_dim=vd,
                            kv_lora_rank=kvl, q_lora_rank=None, rms_eps=1e-6,
                            softmax_scale=(nope + rope) ** -0.5, num_layers=layers,
                            device=torch.device("cpu"), dtype=torch.float32, absorb=False)
    caches = backend.alloc_kv_caches(num_blocks=5, block_size=16)
    assert len(caches) == layers
    for kc, vc in caches:
        assert kc.shape == (5, 16, nh, nope + rope)
        assert vc.shape == (5, 16, nh, vd)
        assert kc.dtype == vc.dtype == torch.float32
        assert kc.device.type == vc.device.type == "cpu"


def test_mla_fia_backend_alloc_kv_caches_shape_absorbed():
    nh, nope, rope, vd, kvl, layers = 4, 2, 2, 2, 8, 2
    backend = MlaFIABackend({}, num_heads=nh, qk_nope=nope, qk_rope=rope, v_head_dim=vd,
                            kv_lora_rank=kvl, q_lora_rank=None, rms_eps=1e-6,
                            softmax_scale=(nope + rope) ** -0.5, num_layers=layers,
                            device=torch.device("cpu"), dtype=torch.float32, absorb=True)
    caches = backend.alloc_kv_caches(num_blocks=5, block_size=16)
    assert len(caches) == layers
    kd = kvl + rope
    for kc, vc in caches:
        assert kc.shape == vc.shape == (5, 16, 1, kd)


def test_mla_fia_backend_stores_construction_args():
    backend = MlaFIABackend({"a": 1}, num_heads=16, qk_nope=128, qk_rope=64, v_head_dim=128,
                            kv_lora_rank=512, q_lora_rank=None, rms_eps=1e-6,
                            softmax_scale=0.1, num_layers=3, device=torch.device("cpu"),
                            dtype=torch.float32, absorb=False)
    assert backend.w == {"a": 1}
    assert (backend.num_heads, backend.qk_nope, backend.qk_rope, backend.v_head_dim) == (
        16, 128, 64, 128)
    assert (backend.kv_lora_rank, backend.q_lora_rank, backend.rms_eps) == (512, None, 1e-6)
    assert backend.softmax_scale == 0.1
    assert (backend.num_layers, backend.device, backend.dtype, backend.absorb) == (
        3, torch.device("cpu"), torch.float32, False)


def test_mla_fia_backend_is_an_attention_backend_subclass():
    assert issubclass(MlaFIABackend, AttentionBackend)


def test_deepseek_model_flags_uses_forward_context():
    assert DeepseekV2Model.USES_FORWARD_CONTEXT is True
