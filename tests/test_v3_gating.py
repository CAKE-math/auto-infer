"""V3 MoE gating (noaux_tc group-limited) shape-level unit test — no weights.
Verifies group-limited selection: chosen experts come only from the top
topk_group groups, exactly top_k of them, finite weights. (§13 risk #8: V3 config
routes correctly through the same model; numerical sign-off is weight-gated.)
"""
import torch

from auto_infer.layers.moe.moe import MoE
from auto_infer.models.deepseek_v2 import DeepseekV2Config

_BASE = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
             kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=4,
             vocab_size=32, n_routed_experts=8, n_shared_experts=1,
             num_experts_per_tok=2, first_k_dense_replace=0)


class _Shim:
    """Minimal carrier for MoE._gate (uses only self.cfg, self.w)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = {}
        self.dtype = torch.float32
    _gate = MoE._gate


def test_v3_group_limited_gating():
    cfg = DeepseekV2Config(dict(_BASE, scoring_func="sigmoid", topk_method="noaux_tc",
                                n_group=4, topk_group=2, norm_topk_prob=True))
    assert cfg.scoring_func == "sigmoid" and cfg.topk_method == "noaux_tc"
    shim = _Shim(cfg)
    T = 3
    router = torch.full((T, 8), -5.0)           # 8 experts = 4 groups of 2
    router[:, [0, 1]] = 5.0                      # group 0 hot
    router[:, [4, 5]] = 4.0                      # group 2 hot ; groups 1,3 cold
    w, idx = shim._gate(router, "model.layers.0.mlp.")
    assert idx.shape == (T, 2) and w.shape == (T, 2)            # exactly top_k
    allowed = {0, 1, 4, 5}                                       # groups 0 and 2 only
    assert set(idx.flatten().tolist()) <= allowed, idx
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-4)  # norm_topk_prob


def test_v2_gating_unchanged():
    """Default config => softmax + greedy top-k (V2-Lite path), no group masking."""
    cfg = DeepseekV2Config(dict(_BASE, norm_topk_prob=False))
    assert cfg.scoring_func == "softmax" and cfg.topk_method == "greedy"
    shim = _Shim(cfg)
    router = torch.randn(4, 8)
    w, idx = shim._gate(router, "model.layers.0.mlp.")
    ref = router.softmax(-1).topk(2, dim=-1)
    assert torch.equal(idx, ref.indices) and torch.allclose(w, ref.values)


if __name__ == "__main__":
    test_v3_group_limited_gating()
    test_v2_gating_unchanged()
    print("ALL PASS: V3 noaux_tc group-limited gating + V2 path unchanged")
