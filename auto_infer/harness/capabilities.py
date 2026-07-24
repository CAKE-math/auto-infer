"""Map inspected checkpoint facts to existing stable runtime capabilities."""


_GQA_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
)
_MLA_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "vocab_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
)


def _missing_config(config: dict, fields) -> list[str]:
    return [
        f"config.{field}" for field in fields
        if config.get(field) is None
    ]


def match_capabilities(manifest: dict) -> dict:
    architecture = manifest["architecture"]
    features = manifest["features"]
    weights = manifest["weights"]
    config = manifest["config"]
    attention = architecture["attention"]
    missing: list[str] = []
    template = None
    entrypoint = None

    if attention in {"mha", "mqa", "gqa"}:
        missing.extend(_missing_config(config, _GQA_CONFIG_FIELDS))
        heads = architecture.get("num_heads")
        kv_heads = architecture.get("num_kv_heads")
        if (isinstance(heads, int) and isinstance(kv_heads, int)
                and kv_heads > 0 and heads % kv_heads):
            missing.append("attention.head_divisibility")
        if features["moe"]:
            missing.append("features.moe")
        if features["sliding_window"]:
            missing.append("features.sliding_window")
        if not weights["standard_layout"]:
            missing.append("weights.standard_layout")
        explicit = config.get("head_dim")
        derived = (
            config["hidden_size"] // config["num_attention_heads"]
            if isinstance(config.get("hidden_size"), int)
            and isinstance(config.get("num_attention_heads"), int)
            and config["num_attention_heads"] else None
        )
        if weights["qk_norm"] or (
            explicit is not None and explicit != derived
        ):
            template = "gqa-qknorm-v1"
            entrypoint = "auto_infer.models.qwen3:Qwen3Model"
        else:
            template = "gqa-swiglu-v1"
            entrypoint = "auto_infer.models.qwen2:Qwen2Model"
    elif attention == "mla":
        missing.extend(_missing_config(config, _MLA_CONFIG_FIELDS))
        if not features["moe"]:
            missing.append("features.moe")
        if features["sliding_window"]:
            missing.append("features.sliding_window")
        if not weights["standard_layout"]:
            missing.append("weights.standard_layout")
        template = "mla-moe-v1"
        entrypoint = "auto_infer.models.deepseek_v2:DeepseekV2Model"
    else:
        missing.extend(_missing_config(config, ("num_attention_heads",)))
        missing.append("attention.supported_family")

    missing = sorted(set(missing))
    if missing:
        entrypoint = None
    if attention in {"mha", "mqa", "gqa"} and not features["moe"]:
        parallel = {
            "tensor": {
                "status": "supported",
                "dtype": "bfloat16",
                "max_size": 8,
                "modes": ["recompute", "paged"],
            },
            "expert": {"status": "unsupported"},
        }
    elif attention == "mla" and features["moe"]:
        parallel = {
            "tensor": {"status": "unsupported"},
            "expert": {"status": "supported", "dtype": "bfloat16"},
        }
    else:
        parallel = {
            "tensor": {"status": "unsupported"},
            "expert": {"status": "unsupported"},
        }
    return {
        "status": "supported" if not missing else "partial",
        "template": template,
        "entrypoint": entrypoint,
        "required": sorted({
            "attention.backend",
            "cache.layout",
            "model.weight_layout",
        }),
        "missing": missing,
        "features": dict(features),
        "parallel": parallel,
    }
