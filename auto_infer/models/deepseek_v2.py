"""DeepSeek-V2 (MLA + MoE + YaRN) for auto-infer.

Bring-up model: DeepSeek-V2-Lite-Chat (27 layers, MLA kv_lora_rank=512,
nope=128/rope=64/v=128, 64 routed + 2 shared experts top-6, first_k_dense=1, YaRN).
Correctness-first: plain-torch MLA attention + naive MoE (runs on NPU, parity vs
HF). Paged MLA FIA + fused-MoE EP are the optimization swaps.
"""
import json
import os

import torch

from auto_infer.layers.mlp import swiglu_mlp
from auto_infer.layers.moe.moe import MoE
from auto_infer.layers.rotary_embedding import build_rope_inv_freq
from auto_infer.models.base import BaseCausalLM


class DeepseekV2Config:
    def __init__(self, d: dict):
        self.hidden_size = d["hidden_size"]
        self.num_layers = d["num_hidden_layers"]
        self.num_heads = d["num_attention_heads"]
        self.kv_lora_rank = d["kv_lora_rank"]
        self.q_lora_rank = d.get("q_lora_rank")
        self.qk_nope = d["qk_nope_head_dim"]
        self.qk_rope = d["qk_rope_head_dim"]
        self.v_head_dim = d["v_head_dim"]
        self.vocab_size = d["vocab_size"]
        self.rms_eps = d.get("rms_norm_eps", 1e-6)
        self.rope_theta = d.get("rope_theta", 10000)
        self.n_routed = d["n_routed_experts"]
        self.n_shared = d["n_shared_experts"]
        self.top_k = d["num_experts_per_tok"]
        self.first_k_dense = d.get("first_k_dense_replace", 0)
        self.norm_topk_prob = d.get("norm_topk_prob", False)
        self.scoring_func = d.get("scoring_func", "softmax")
        self.topk_method = d.get("topk_method", "greedy")
        self.n_group = d.get("n_group", 1)
        self.topk_group = d.get("topk_group", 1)
        self.routed_scale = d.get("routed_scaling_factor", 1.0)
        self.rope_scaling = d.get("rope_scaling")
        self.tie = d.get("tie_word_embeddings", False)

    @classmethod
    def from_path(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            return cls(json.load(f))


class DeepseekV2Model(BaseCausalLM):
    ATTENTION_FAMILY = "mla"

    def __init__(self, config, device, dtype):
        self.cfg = config
        self.device = device
        self.dtype = dtype
        self.w: dict[str, torch.Tensor] = {}
        self.moe: MoE | None = None   # built in from_pretrained once self.w is loaded
        self._fused_moe = True      # merge fused grouped-GEMM MoE as default (ep=1)
        self._fused_free = True     # drop per-expert originals after stacking (no 2x mem)
        self._mla_absorb = False    # absorbed MLA verified (math) but this CANN FIA
        #                             rejects latent head-dim 576 (only 64/128/192); opt-in for
        #                             a newer MLA op / custom kernel. Non-absorbed FIA is default.
        self._inv_freq = None
        self._cs_mscale = 1.0
        self._softmax_scale = (config.qk_nope + config.qk_rope) ** -0.5
        self._init_rope()

    def _init_rope(self):
        cfg = self.cfg
        self._inv_freq, self._cs_mscale, ss_mult = build_rope_inv_freq(
            cfg.qk_rope, cfg.rope_theta, cfg.rope_scaling, self.device)
        self._softmax_scale *= ss_mult

    @classmethod
    def from_pretrained(cls, path, device, dtype=torch.bfloat16, ep_size=1, ep_rank=0):
        """Load weights via the sharded loader: per-tensor streaming reads, never
        materializing the full state dict. When ep_size>1 each rank reads ONLY its
        EP slice of the routed experts (must match the ep_size/ep_rank the MoE
        forward uses), so 671B never OOMs a single host."""
        from auto_infer.models.loader import (expert_shard_predicate, load_sharded,
                                               start_prefetch)
        start_prefetch(path)  # warm page cache in background while cfg/m init below runs
        cfg = DeepseekV2Config.from_path(path)
        m = cls(cfg, device, dtype)
        wanted = expert_shard_predicate(cfg.n_routed, ep_size, ep_rank)
        m.w = load_sharded(path, wanted, device=device, dtype=dtype)
        if "lm_head.weight" not in m.w:
            m.w["lm_head.weight"] = m.w["model.embed_tokens.weight"]
        m.moe = MoE(m.w, cfg, device, dtype, fused=m._fused_moe,
                    free_originals=m._fused_free, layer_prefix=m.layer_prefix)
        return m

    def _cos_sin(self, positions):
        freqs = torch.outer(positions.float(), self._inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self._cs_mscale).to(self.dtype), (emb.sin() * self._cs_mscale).to(self.dtype)

    _compute_cos_sin = _cos_sin   # BaseCausalLM rope hook

    def _ffn(self, i, x, prefix, ctx):
        return self.moe(x, i, ctx.active_token_mask) if i >= self.cfg.first_k_dense \
            else swiglu_mlp(x, self.w, prefix + "mlp.")
