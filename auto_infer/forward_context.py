"""Neutral model/attention execution contract."""
from dataclasses import dataclass

import torch

from auto_infer.layers.attention.base import AttentionBackend


@dataclass
class ForwardContext:
    token_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: list
    seqlens_kv: list
    attn_mask: torch.Tensor
    attn_backend: AttentionBackend
    kv_caches: list
    is_decode: bool
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
    active_token_mask: torch.Tensor | None = None
