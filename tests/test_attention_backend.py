"""Host-only unit tests for layers/attention/backend.py (SP1 task 1; updated
for SP5's generalized `attention(layer_idx, x, ctx)` seam — see
docs/superpowers/specs/2026-07-17-skeleton-sp5-deepseek-mla-backend.md).

`DenseBackend`'s core attention math is checked against a hand-computed
causal-softmax reference (no torch_npu needed) via its private `_attn` core
(the part independent of proj/RoPE/o_proj). `GqaFIABackend`/`GraphGqaBackend`
are only checked for their pure-Python surface (construction + KV-cache
allocation shape/dtype/device); their `_write_kv`/`_attn` call straight into
torch_npu ops and are exercised by the NPU-only per-layer parity harness
(tools/parity.py), not here.
"""
import pytest
import torch
from types import SimpleNamespace

from auto_infer.layers.attention.base import AttentionBackend
from auto_infer.layers.attention.gqa import (
    DenseBackend, GqaFIABackend, GraphGqaBackend)
from auto_infer.layers.attention.mla import MlaDenseBackend
from auto_infer.layers.attention.registry import (
    build_attention_backend,
    build_mtp_attention_backend,
    register_attention_family,
    register_mtp_attention_family,
)
from auto_infer.models.qwen2 import pack_qwen2_projections


def test_attention_backend_is_abstract():
    with pytest.raises(TypeError):
        AttentionBackend()


def _ref_causal_attn(q, k, v, scale):
    """Tiny hand-computed reference: single-head, single-request causal softmax
    attention, fp32 throughout. q/k/v: (T, hd)."""
    T = q.shape[0]
    scores = (q.float() @ k.float().t()) * scale
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    scores = scores + causal
    attn = torch.softmax(scores, dim=-1)
    return attn @ v.float()


def test_dense_backend_core_matches_hand_reference_single_head():
    torch.manual_seed(0)
    T, hd = 4, 8
    n_q = n_kv = 1
    scale = hd ** -0.5
    q = torch.randn(T, n_q, hd)
    k = torch.randn(T, n_kv, hd)
    v = torch.randn(T, n_kv, hd)

    backend = DenseBackend(n_q_heads=n_q, n_kv_heads=n_kv, head_dim=hd, scale=scale)

    class _Ctx:
        cu_seqlens_q = [T]

    out = backend._attn(0, q, k, v, _Ctx())
    assert out.shape == (T, n_q, hd)

    expected = _ref_causal_attn(q[:, 0], k[:, 0], v[:, 0], scale)
    torch.testing.assert_close(out[:, 0].float(), expected, atol=1e-5, rtol=1e-5)


def test_dense_backend_core_gqa_repeat_interleave_matches_reference():
    """n_q_heads = 2 * n_kv_heads: each KV head must be shared by its 2 query
    heads via repeat_interleave (GQA), matching a per-head hand reference."""
    torch.manual_seed(1)
    T, hd = 5, 4
    n_kv, n_q = 2, 4
    scale = hd ** -0.5
    q = torch.randn(T, n_q, hd)
    k = torch.randn(T, n_kv, hd)
    v = torch.randn(T, n_kv, hd)

    backend = DenseBackend(n_q_heads=n_q, n_kv_heads=n_kv, head_dim=hd, scale=scale)

    class _Ctx:
        cu_seqlens_q = [T]

    out = backend._attn(0, q, k, v, _Ctx())
    assert out.shape == (T, n_q, hd)

    rep = n_q // n_kv
    for qh in range(n_q):
        kvh = qh // rep
        expected = _ref_causal_attn(q[:, qh], k[:, kvh], v[:, kvh], scale)
        torch.testing.assert_close(out[:, qh].float(), expected, atol=1e-5, rtol=1e-5)


def test_dense_backend_core_respects_per_request_segments():
    """Two concatenated requests (TND layout) must NOT attend across the
    request boundary — segment 2 should be identical whether or not segment 1
    is prepended."""
    torch.manual_seed(2)
    hd = 4
    n_q = n_kv = 1
    scale = hd ** -0.5
    t1, t2 = 3, 2
    q_all = torch.randn(t1 + t2, n_q, hd)
    k_all = torch.randn(t1 + t2, n_kv, hd)
    v_all = torch.randn(t1 + t2, n_kv, hd)

    backend = DenseBackend(n_q_heads=n_q, n_kv_heads=n_kv, head_dim=hd, scale=scale)

    class _Ctx:
        cu_seqlens_q = [t1, t1 + t2]

    out = backend._attn(0, q_all, k_all, v_all, _Ctx())

    # second segment computed in isolation must match the batched result.
    q2, k2, v2 = q_all[t1:], k_all[t1:], v_all[t1:]

    class _Ctx2:
        cu_seqlens_q = [t2]

    out2 = backend._attn(0, q2, k2, v2, _Ctx2())
    torch.testing.assert_close(out[t1:], out2, atol=1e-6, rtol=1e-6)


def test_mla_dense_backend_respects_per_request_segments():
    torch.manual_seed(4)
    t1, t2, heads, qk, vd = 3, 2, 2, 4, 3
    backend = MlaDenseBackend(
        {}, num_heads=heads, qk_nope=2, qk_rope=2,
        v_head_dim=vd, kv_lora_rank=2, q_lora_rank=None,
        rms_eps=1e-6, softmax_scale=qk ** -0.5)
    q = torch.randn(t1 + t2, heads, qk)
    k = torch.randn(t1 + t2, heads, qk)
    v = torch.randn(t1 + t2, heads, vd)

    packed = backend._attn(
        0, q, k, v, SimpleNamespace(cu_seqlens_q=[t1, t1 + t2]))
    isolated = backend._attn(
        0, q[t1:], k[t1:], v[t1:],
        SimpleNamespace(cu_seqlens_q=[t2]))

    torch.testing.assert_close(packed[t1:], isolated)


def test_dense_backend_alloc_and_write_kv_are_noops():
    backend = DenseBackend(n_q_heads=2, n_kv_heads=1, head_dim=4, scale=1.0)
    assert backend.alloc_kv_caches(8, 16) == []
    assert backend._write_kv(0, torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), ctx=None) is None


def test_gqa_fia_backend_alloc_kv_caches_shape_dtype_device():
    n_kv, hd, num_layers = 4, 16, 3
    backend = GqaFIABackend(n_q_heads=8, n_kv_heads=n_kv, head_dim=hd, scale=hd ** -0.5,
                            num_layers=num_layers, device=torch.device("cpu"),
                            dtype=torch.float32)
    caches = backend.alloc_kv_caches(num_blocks=10, block_size=32)
    assert len(caches) == num_layers
    for c in caches:
        assert c.shape == (2, 10, 32, n_kv, hd)
        assert c.dtype == torch.float32
        assert c.device.type == "cpu"


def test_gqa_fia_backend_stores_construction_args():
    backend = GqaFIABackend(n_q_heads=16, n_kv_heads=2, head_dim=64, scale=0.125,
                            num_layers=3, device=torch.device("cpu"), dtype=torch.float32)
    assert (backend.n_q_heads, backend.n_kv_heads, backend.head_dim, backend.scale) == (
        16, 2, 64, 0.125)
    assert (backend.num_layers, backend.device, backend.dtype) == (
        3, torch.device("cpu"), torch.float32)


def test_gqa_fia_backend_is_an_attention_backend_subclass():
    assert issubclass(GqaFIABackend, AttentionBackend)


def test_dense_backend_is_an_attention_backend_subclass():
    assert issubclass(DenseBackend, AttentionBackend)


def test_attention_backend_requires_attention_and_alloc_kv_caches():
    """The ABC's abstract surface is exactly `alloc_kv_caches` + `attention`
    (SP5: generalized from the old `write_kv`/`attn` q/k/v-in seam)."""
    assert AttentionBackend.__abstractmethods__ == frozenset(
        {"alloc_kv_caches", "attention"})


def test_registered_attention_family_needs_no_dispatcher_branch():
    calls = []

    class SyntheticBackend:
        def alloc_kv_caches(self, num_blocks, block_size):
            calls.append((num_blocks, block_size))
            return ["cache"]

    register_attention_family(
        "synthetic-test",
        lambda model, mode: SyntheticBackend(),
    )

    class SyntheticModel:
        ATTENTION_FAMILY = "synthetic-test"

    backend, caches = build_attention_backend(
        SyntheticModel(), "graph", num_blocks=3, block_size=16)

    assert isinstance(backend, SyntheticBackend)
    assert caches == ["cache"]
    assert calls == [(3, 16)]


def test_attention_registry_rejects_duplicate_names():
    builder = lambda model, mode: None
    register_attention_family("duplicate-attention-test", builder)
    with pytest.raises(ValueError, match="already registered"):
        register_attention_family("duplicate-attention-test", builder)


def test_mtp_attention_capability_is_registered_separately():
    calls = []

    class SyntheticBackend:
        def alloc_kv_caches(self, num_blocks, block_size):
            calls.append((num_blocks, block_size))
            return ["mtp-cache"]

    register_mtp_attention_family(
        "synthetic-mtp-test",
        lambda model, mode, prefix: SyntheticBackend(),
    )

    class SyntheticModel:
        ATTENTION_FAMILY = "synthetic-mtp-test"

    backend, caches = build_mtp_attention_backend(
        SyntheticModel(), "graph", "model.mtp.", 5, 16)

    assert isinstance(backend, SyntheticBackend)
    assert caches == ["mtp-cache"]
    assert calls == [(5, 16)]


def test_mla_mtp_capability_fails_explicitly_until_implemented():
    model = SimpleNamespace(ATTENTION_FAMILY="mla")

    with pytest.raises(NotImplementedError, match="mla.*MTP.*graph"):
        build_mtp_attention_backend(model, "graph", "model.mtp.", 5, 16)


@pytest.mark.parametrize(
    ("mode", "backend_type"),
    (("paged", GqaFIABackend), ("graph", GraphGqaBackend)),
)
def test_gqa_mtp_capability_builds_one_layer_with_requested_prefix(
        mode, backend_type):
    model = SimpleNamespace(
        ATTENTION_FAMILY="gqa",
        cfg=SimpleNamespace(head_dim=16, rms_eps=1e-6),
        n_q_local=2,
        n_kv_local=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        w={},
    )

    backend, caches = build_mtp_attention_backend(
        model, mode, "model.mtp_layers.0.", 5, 16)

    assert isinstance(backend, backend_type)
    assert backend.num_layers == 1
    assert backend.layer_prefix(0) == "model.mtp_layers.0."
    assert len(caches) == 1


def test_dense_backend_attention_matches_manual_proj_rope_core_o_proj():
    """End-to-end check of the shared `_GqaProjRopeMixin.attention` wrapper
    (proj -> RoPE -> core attn -> reshape -> o_proj) against the same math done
    by hand, using `DenseBackend`'s full-softmax core. Weights are read from the
    backend's `w` dict by name via the model-owned `layer_prefix`."""
    torch.manual_seed(3)
    T, hidden, n_q, n_kv, hd = 4, 8, 2, 1, 4
    scale = hd ** -0.5
    x = torch.randn(T, hidden)
    p = "model.layers.0.self_attn."
    w = {
        p + "q_proj.weight": torch.randn(n_q * hd, hidden), p + "q_proj.bias": torch.randn(n_q * hd),
        p + "k_proj.weight": torch.randn(n_kv * hd, hidden), p + "k_proj.bias": torch.randn(n_kv * hd),
        p + "v_proj.weight": torch.randn(n_kv * hd, hidden), p + "v_proj.bias": torch.randn(n_kv * hd),
        p + "o_proj.weight": torch.randn(hidden, n_q * hd),
    }
    original = dict(w)
    # Production Qwen weights are packed once, after TP sharding.
    pack_qwen2_projections(w, num_layers=1)
    cos = torch.randn(T, 1, hd)
    sin = torch.randn(T, 1, hd)

    class _Ctx:
        cu_seqlens_q = [T]

    ctx = _Ctx()
    ctx.cos, ctx.sin = cos, sin

    backend = DenseBackend(n_q_heads=n_q, n_kv_heads=n_kv, head_dim=hd, scale=scale, w=w)
    out = backend.attention(0, x, ctx)
    assert out.shape == (T, hidden)

    def _rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    q = (x @ original[p + "q_proj.weight"].t() + original[p + "q_proj.bias"]).view(T, n_q, hd)
    k = (x @ original[p + "k_proj.weight"].t() + original[p + "k_proj.bias"]).view(T, n_kv, hd)
    v = (x @ original[p + "v_proj.weight"].t() + original[p + "v_proj.bias"]).view(T, n_kv, hd)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    core = backend._attn(0, q, k, v, ctx).reshape(T, n_q * hd)
    expected = core @ w[p + "o_proj.weight"].t()
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
