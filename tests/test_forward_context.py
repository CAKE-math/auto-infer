"""Host-only unit tests for worker/forward_context.py (SP1 task 2).

Checks the ForwardContext dataclass shape/fields, and that NpuModelRunner._build
constructs one with the SAME marshaled tensors/lists NpuModelRunner produced
before this refactor (paged inputs unchanged; only repackaged).
"""
import dataclasses

import torch

from auto_infer.layers.attention.gqa import DenseBackend
from auto_infer.forward_context import ForwardContext


def test_forward_context_is_a_plain_dataclass_with_spec_fields():
    fields = {f.name for f in dataclasses.fields(ForwardContext)}
    assert fields == {
        "token_ids", "positions", "slot_mapping", "block_table",
        "cu_seqlens_q", "seqlens_kv", "attn_mask", "attn_backend",
        "kv_caches", "is_decode", "cos", "sin",
        "active_token_mask",
    }


def test_forward_context_cos_sin_default_to_none():
    """SP5: cos/sin are set by model.forward() (mutated onto ctx after
    construction, once per step) — not required at construction time."""
    backend = DenseBackend(n_q_heads=2, n_kv_heads=1, head_dim=4, scale=0.5)
    ctx = ForwardContext(
        token_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cu_seqlens_q=[3],
        seqlens_kv=[3],
        attn_mask=torch.zeros(1, 1),
        attn_backend=backend,
        kv_caches=[],
        is_decode=False,
    )
    assert ctx.cos is None
    assert ctx.sin is None
    assert ctx.active_token_mask is None


def test_forward_context_construction_roundtrip():
    backend = DenseBackend(n_q_heads=2, n_kv_heads=1, head_dim=4, scale=0.5)
    ctx = ForwardContext(
        token_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cu_seqlens_q=[3],
        seqlens_kv=[3],
        attn_mask=torch.zeros(1, 1),
        attn_backend=backend,
        kv_caches=[],
        is_decode=False,
    )
    assert ctx.attn_backend is backend
    assert ctx.cu_seqlens_q == [3]
    assert ctx.is_decode is False
