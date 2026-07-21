"""Qwen3 dense model for auto-infer.

Qwen3 dense = Qwen2's GQA decoder with three deltas, ALL data-driven in the
shared `_GqaProjRopeMixin` / model shell (so `Qwen3Model` is just `Qwen2Model`
with a different config class — a demonstration of the skeleton's fast-adapt
goal):

  1. QK-Norm — q and k get a per-head RMSNorm (weights
     `self_attn.q_norm`/`k_norm`, shape `(head_dim,)`) AFTER the q/k/v
     projection and BEFORE RoPE. The mixin applies it whenever a layer's
     `q_norm.weight` is present in `w`; Qwen2 (no such weight) skips it.
  2. No attention bias — `attention_bias=false`; q/k/v projections have no
     bias. The mixin reads bias via `w.get(...)`, so a missing key is just None.
  3. Independent `head_dim` — Qwen3 sets `head_dim` explicitly (e.g. 128) rather
     than `hidden_size // num_heads` (Qwen3-0.6B: hidden 1024, 16 heads, but
     head_dim 128 => q_proj is 2048-wide). `Qwen3Config` reads it from config.
"""
from auto_infer.models.qwen2 import Qwen2Config, Qwen2Model


class Qwen3Config(Qwen2Config):
    def __init__(self, d: dict):
        super().__init__(d)
        self.head_dim = d["head_dim"]                 # Qwen3: explicit (not hidden//heads)


class Qwen3Model(Qwen2Model):
    """See module docstring — the three Qwen3 deltas are handled by the shared
    backend/mixin + config, so no forward/backend overrides are needed."""
    _CONFIG_CLS = Qwen3Config
