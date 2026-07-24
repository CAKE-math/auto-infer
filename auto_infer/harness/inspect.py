"""Read checkpoint facts without loading tensors or importing model code."""

import json
from pathlib import Path

from auto_infer.harness.artifacts import sha256_file


_GQA_WEIGHT_KEYS = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
}
_MLA_WEIGHT_KEYS = {
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.kv_b_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
}


def _weight_keys(model_path: Path) -> tuple[set[str], str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        return set(payload.get("weight_map", {})), "index"
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        return set(), "absent"
    try:
        from safetensors import safe_open
        keys = set()
        for path in files:
            with safe_open(path, framework="pt") as source:
                keys.update(source.keys())
        return keys, "headers"
    except (ImportError, OSError, ValueError):
        return set(), "unreadable"


def _attention(config: dict) -> str:
    mla_fields = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    if all(config.get(field) is not None for field in mla_fields):
        return "mla"
    heads = config.get("num_attention_heads")
    if not isinstance(heads, int) or heads <= 0:
        return "unknown"
    kv_heads = config.get("num_key_value_heads", heads)
    if not isinstance(kv_heads, int) or kv_heads <= 0:
        return "unknown"
    if kv_heads == heads:
        return "mha"
    if kv_heads == 1:
        return "mqa"
    if kv_heads < heads:
        return "gqa"
    return "unknown"


def _has_moe(config: dict) -> bool:
    return any(
        isinstance(config.get(field), int) and config[field] > 0
        for field in ("n_routed_experts", "num_local_experts", "num_experts")
    )


def _has_mtp(config: dict, keys: set[str]) -> bool:
    return (
        any(key.startswith("model.mtp_layers.") for key in keys)
        or any(
            isinstance(config.get(field), int) and config[field] > 0
            for field in ("num_nextn_predict_layers", "num_mtp_layers")
        )
    )


def _standard_layout(attention: str, config: dict,
                     keys: set[str]) -> tuple[bool, list[str]]:
    if attention in {"mha", "mqa", "gqa"}:
        required = _GQA_WEIGHT_KEYS
    elif attention == "mla":
        required = set(_MLA_WEIGHT_KEYS)
        if config.get("q_lora_rank") is None:
            required.add("model.layers.0.self_attn.q_proj.weight")
        else:
            required.update({
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.0.self_attn.q_b_proj.weight",
            })
    else:
        return False, []
    missing = sorted(required - keys)
    return not missing, missing


def inspect_model(model_path: Path) -> dict:
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    architectures = config.get("architectures") or []
    architecture_name = architectures[0] if architectures else None
    attention = _attention(config)
    heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    hidden = config.get("hidden_size")
    explicit_head_dim = config.get("head_dim")
    head_dim = explicit_head_dim
    if head_dim is None and isinstance(hidden, int) and isinstance(heads, int) and heads:
        head_dim = hidden // heads
    keys, evidence = _weight_keys(model_path)
    standard_layout, missing_weight_keys = _standard_layout(
        attention, config, keys)
    qk_norm = {
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    }.issubset(keys)
    position = (
        "yarn" if isinstance(config.get("rope_scaling"), dict)
        and config["rope_scaling"].get("type") == "yarn"
        else "rope"
    )
    return {
        "schema_version": 1,
        "source": {
            "model_path": str(model_path),
            "config_sha256": sha256_file(config_path),
            "architecture": architecture_name,
            "model_type": config.get("model_type"),
        },
        "architecture": {
            "type": "decoder_only",
            "attention": attention,
            "num_layers": config.get("num_hidden_layers"),
            "hidden_size": hidden,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "position_embedding": position,
        },
        "features": {
            "moe": _has_moe(config),
            "mtp": _has_mtp(config, keys),
            "sliding_window": bool(config.get("sliding_window")),
        },
        "cache": {
            "type": "paged_latent" if attention == "mla" else "paged_kv",
            "kv_lora_rank": config.get("kv_lora_rank"),
            "qk_nope_head_dim": config.get("qk_nope_head_dim"),
            "qk_rope_head_dim": config.get("qk_rope_head_dim"),
            "v_head_dim": config.get("v_head_dim"),
        },
        "config": config,
        "weights": {
            "evidence": evidence,
            "key_count": len(keys),
            "standard_layout": standard_layout,
            "missing_standard_keys": missing_weight_keys,
            "qk_norm": qk_norm,
        },
    }

