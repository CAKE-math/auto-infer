"""Registered attention-family construction independent of model subclasses."""
from typing import Callable


AttentionBuilder = Callable[[object, str], object]
MtpAttentionBuilder = Callable[[object, str, str], object]
_FAMILIES: dict[str, AttentionBuilder] = {}
_MTP_FAMILIES: dict[str, MtpAttentionBuilder] = {}


def register_attention_family(name: str, builder: AttentionBuilder) -> None:
    if not name:
        raise ValueError("attention family name must not be empty")
    if name in _FAMILIES:
        raise ValueError(f"attention family already registered: {name}")
    _FAMILIES[name] = builder


def register_mtp_attention_family(
    name: str,
    builder: MtpAttentionBuilder,
) -> None:
    """Register a family only after its recurrent MTP math is supported."""
    if not name:
        raise ValueError("MTP attention family name must not be empty")
    if name in _MTP_FAMILIES:
        raise ValueError(f"MTP attention family already registered: {name}")
    _MTP_FAMILIES[name] = builder


def _gqa(model, mode: str):
    from auto_infer.layers.attention.gqa import (
        DenseBackend,
        GqaFIABackend,
        GraphGqaBackend,
    )
    cfg = model.cfg
    common = dict(
        n_q_heads=model.n_q_local,
        n_kv_heads=model.n_kv_local,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim ** -0.5,
        w=model.w,
        layer_prefix=model.layer_prefix,
        rms_eps=cfg.rms_eps,
    )
    if mode == "dense":
        return DenseBackend(**common)
    backend = GqaFIABackend if mode == "paged" else GraphGqaBackend
    return backend(
        num_layers=cfg.num_layers,
        device=model.device,
        dtype=model.dtype,
        **common,
    )


def _mla(model, mode: str):
    from auto_infer.layers.attention.mla import (
        GraphMlaBackend,
        MlaDenseBackend,
        MlaFIABackend,
    )
    cfg = model.cfg
    common = dict(
        w=model.w,
        num_heads=cfg.num_heads,
        qk_nope=cfg.qk_nope,
        qk_rope=cfg.qk_rope,
        v_head_dim=cfg.v_head_dim,
        kv_lora_rank=cfg.kv_lora_rank,
        q_lora_rank=cfg.q_lora_rank,
        rms_eps=cfg.rms_eps,
        softmax_scale=model._softmax_scale,
        layer_prefix=model.layer_prefix,
    )
    if mode == "dense":
        return MlaDenseBackend(**common)
    if mode == "paged":
        return MlaFIABackend(
            num_layers=cfg.num_layers,
            device=model.device,
            dtype=model.dtype,
            absorb=model._mla_absorb,
            **common,
        )
    return GraphMlaBackend(
        num_layers=cfg.num_layers,
        device=model.device,
        dtype=model.dtype,
        **common,
    )


def _gqa_mtp(model, mode: str, prefix: str):
    from auto_infer.layers.attention.gqa import GqaFIABackend, GraphGqaBackend

    cfg = model.cfg
    backend = GqaFIABackend if mode == "paged" else GraphGqaBackend
    return backend(
        n_q_heads=model.n_q_local,
        n_kv_heads=model.n_kv_local,
        head_dim=cfg.head_dim,
        scale=cfg.head_dim ** -0.5,
        num_layers=1,
        device=model.device,
        dtype=model.dtype,
        w=model.w,
        layer_prefix=lambda _: prefix,
        rms_eps=cfg.rms_eps,
    )


register_attention_family("gqa", _gqa)
register_attention_family("mla", _mla)
register_mtp_attention_family("gqa", _gqa_mtp)


def build_attention_backend(
    model,
    mode: str,
    num_blocks: int = 0,
    block_size: int = 0,
):
    if mode not in {"dense", "paged", "graph"}:
        raise ValueError(f"unsupported attention mode: {mode}")
    family = model.ATTENTION_FAMILY
    try:
        builder = _FAMILIES[family]
    except KeyError:
        choices = ", ".join(sorted(_FAMILIES))
        raise KeyError(
            f"unregistered attention family: {family}; registered families: {choices}"
        ) from None
    backend = builder(model, mode)
    return backend, backend.alloc_kv_caches(num_blocks, block_size)


def build_mtp_attention_backend(
    model,
    mode: str,
    prefix: str,
    num_blocks: int = 0,
    block_size: int = 0,
):
    """Build the separately supported recurrent-MTP attention capability."""
    if mode not in {"paged", "graph"}:
        raise ValueError(f"unsupported MTP attention mode: {mode}")
    family = model.ATTENTION_FAMILY
    try:
        builder = _MTP_FAMILIES[family]
    except KeyError:
        raise NotImplementedError(
            f"{family} MTP attention does not support {mode} mode"
        ) from None
    backend = builder(model, mode, prefix)
    return backend, backend.alloc_kv_caches(num_blocks, block_size)
