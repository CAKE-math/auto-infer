"""Qwen2 dense GQA model for auto-infer (NPU). Also the loader for Qwen3 (via
`Qwen3Model` subclass) and MiMo (registered directly — MiMo base == Qwen2 dense).

Runs the shared `BaseCausalLM.forward(ctx)` with an injected `AttentionBackend`:
`GqaFIABackend` (eager paged FIA), `DenseBackend` (bring-up/parity, full-softmax),
`GraphGqaBackend` (ACL-graph decode) — selected by the attention registry.
`forward_cp` is the separate
context-parallel prefill path. Weights stream in via the sharded loader
(`from_pretrained`); W8A8 is an opt-in quantize pass.
"""
import json
import os

import torch

from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.layers.rotary_embedding import rotate_half as _rotate_half
from auto_infer.models.base import BaseCausalLM


def _concat_output_projections(values):
    if isinstance(values[0], tuple):
        if not all(isinstance(value, tuple) for value in values):
            raise TypeError("cannot mix floating and quantized projections")
        return (
            torch.cat([value[0] for value in values], dim=1).contiguous(),
            torch.cat([value[1] for value in values], dim=0).contiguous(),
        )
    return torch.cat(values, dim=0).contiguous()


def pack_qwen2_projections(w: dict, num_layers: int) -> dict:
    """Pack column-parallel projections after TP sharding.

    Keeping the packed tensors in the canonical weight dictionary makes every
    execution backend use the same one-GEMM QKV and gate/up representation.
    Source tensors are removed so model memory does not grow.
    """
    layers = [f"model.layers.{i}." for i in range(num_layers)]
    mtp_suffix = "self_attn.q_proj.weight"
    layers.extend(sorted({
        name[:-len(mtp_suffix)]
        for name in w
        if name.startswith("model.mtp_layers.") and name.endswith(mtp_suffix)
    }))
    for layer in layers:
        attn = layer + "self_attn."
        qkv_key = attn + "qkv_proj.weight"
        if qkv_key not in w and attn + "q_proj.weight" in w:
            names = [attn + f"{name}_proj.weight" for name in ("q", "k", "v")]
            w[qkv_key] = _concat_output_projections([w.pop(name) for name in names])
            bias_names = [attn + f"{name}_proj.bias" for name in ("q", "k", "v")]
            present = [name in w for name in bias_names]
            if any(present):
                if not all(present):
                    raise ValueError(f"partial QKV bias set in {layer}")
                w[attn + "qkv_proj.bias"] = torch.cat(
                    [w.pop(name) for name in bias_names], dim=0).contiguous()

        mlp = layer + "mlp."
        gate_up_key = mlp + "gate_up_proj.weight"
        if gate_up_key not in w and mlp + "gate_proj.weight" in w:
            w[gate_up_key] = _concat_output_projections([
                w.pop(mlp + "gate_proj.weight"),
                w.pop(mlp + "up_proj.weight"),
            ])
    return w


class Qwen2Config:
    def __init__(self, d: dict):
        self.hidden_size = d["hidden_size"]
        self.num_layers = d["num_hidden_layers"]
        self.num_heads = d["num_attention_heads"]
        self.num_kv_heads = d["num_key_value_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = d["intermediate_size"]
        self.vocab_size = d["vocab_size"]
        self.rms_eps = d.get("rms_norm_eps", 1e-6)
        self.rope_theta = d.get("rope_theta", 1000000.0)
        self.tie_word_embeddings = d.get("tie_word_embeddings", True)

    @classmethod
    def from_path(cls, path: str) -> "Qwen2Config":
        with open(os.path.join(path, "config.json")) as f:
            return cls(json.load(f))


class Qwen2Model(BaseCausalLM):
    ATTENTION_FAMILY = "gqa"
    SUPPORTS_TENSOR_PARALLEL = True
    _CONFIG_CLS = Qwen2Config     # subclasses (Qwen3Model) override to parse their config

    def __init__(self, config: Qwen2Config, device: torch.device, dtype: torch.dtype):
        self.cfg = config
        self.device = device
        self.dtype = dtype
        self.w: dict[str, torch.Tensor] = {}
        self.tp_rank, self.tp_size = 0, 1
        self.n_q_local = config.num_heads
        self.n_kv_local = config.num_kv_heads
        self.quant: str | None = None
        self.use_custom_kernels = False   # route SwiGLU through Triton-Ascend kernel

    def _swiglu(self, gate, up):
        if self.use_custom_kernels:
            from auto_infer.layers.kernels.swiglu_triton import silu_mul
            return silu_mul(gate, up)
        import torch_npu
        return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1))    # fused CANN SwiGLU (§5)

    def _lin(self, x, wname, bias=None):
        """W8A8-aware linear by weight name — shared dispatch lives in
        layers/attention/base._lin (single source of truth)."""
        from auto_infer.layers.attention.base import _lin
        return _lin(x, self.w[wname], bias=bias)

    def prepare_packed_projections(self) -> None:
        pack_qwen2_projections(self.w, self.cfg.num_layers)

    @classmethod
    def from_pretrained(cls, path: str, device: torch.device,
                        dtype: torch.dtype = torch.bfloat16,
                        tp_rank: int = 0, tp_size: int = 1,
                        quantize: str | None = None, **_) -> "Qwen2Model":
        """Weights loaded via the streaming sharded loader (spec §15.1), aligned
        with the DeepSeek path (`models/deepseek_v2.py`): `start_prefetch`
        launches a background page-cache warm for the checkpoint's shard files
        FIRST (overlapping disk I/O with `Qwen2Config.from_path` + model
        construction below), then `load_sharded` reads each tensor straight to
        `device`/`dtype` via `safe_open`/`get_tensor` (parallel across shard
        files, `max_workers=8`) — never materializing a whole shard file or a
        full eager state dict on host RAM the way the old
        `safetensors.torch.load_file` loop did. Behavior-preserving: Qwen2
        checkpoints are all-floating-point (no int/quant tensors pre-quantize),
        so `load_sharded`'s "cast floats to dtype, leave non-float untouched"
        is equivalent to the old unconditional `.to(device=device, dtype=dtype)`
        (see `tests/test_weight_loader.py::test_qwen2_new_loader_matches_old`).
        """
        from auto_infer.models.loader import load_sharded, start_prefetch
        from auto_infer.models.parallel import TensorParallelPlan
        cfg = cls._CONFIG_CLS.from_path(path)
        plan = TensorParallelPlan.for_qwen(cfg, tp_rank, tp_size)
        model = cls(cfg, device, dtype)
        model.tp_rank, model.tp_size = tp_rank, tp_size
        model.quant = quantize
        model.n_q_local = plan.q_rows // cfg.head_dim
        model.n_kv_local = plan.kv_rows // cfg.head_dim
        start_prefetch(path)
        w = load_sharded(path, wanted=lambda n: True, device=device, dtype=dtype,
                          max_workers=8,
                          slicer=(plan.slice_spec if tp_size > 1 else None))
        # The model configuration, rather than redundant checkpoint storage,
        # owns the tying contract. Some checkpoints serialize both tensors
        # even when they are tied; retaining both wastes one vocabulary-sized
        # matrix and can silently use a stale copy after conversion.
        if cfg.tie_word_embeddings or "lm_head.weight" not in w:
            w["lm_head.weight"] = w["model.embed_tokens.weight"]
        if quantize == "w8a8":
            from auto_infer.layers.quantization.w8a8 import quantize_weight
            for i in range(cfg.num_layers):
                for nm in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                    key = f"model.layers.{i}.{nm}.weight"
                    w[key] = quantize_weight(w[key])   # -> (w_int8_t, w_scale)
        model.w = w
        model.prepare_packed_projections()
        return model

    def _rope_cos_sin(self, positions: torch.Tensor):
        hd = self.cfg.head_dim
        inv_freq = 1.0 / (self.cfg.rope_theta ** (
            torch.arange(0, hd, 2, device=self.device, dtype=torch.float32) / hd))
        freqs = torch.outer(positions.float(), inv_freq)          # (T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)                   # (T, hd)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    _compute_cos_sin = _rope_cos_sin   # BaseCausalLM rope hook

    @torch.no_grad()
    def forward_cp(self, local_ids, local_positions, full_T, cp_rank, cp_size):
        """Context-parallel prefill: this rank holds sequence chunk [start:start+lt].
        Per layer, all-gather K/V across CP ranks; local Q attends to full K/V with
        chunked causal mask. Returns local-chunk logits (lt, vocab)."""
        from auto_infer.distributed.parallel_state import cp_all_gather
        cfg, w = self.cfg, self.w
        n_q, n_kv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        scale = hd ** -0.5
        lt = local_ids.shape[0]
        start = cp_rank * lt
        dev = self.device
        h = w["model.embed_tokens.weight"][local_ids]
        cos, sin = self._rope_cos_sin(local_positions)
        cos_h, sin_h = cos.unsqueeze(1), sin.unsqueeze(1)
        abs_q = torch.arange(start, start + lt, device=dev).unsqueeze(1)   # (lt,1)
        kv_idx = torch.arange(full_T, device=dev).unsqueeze(0)            # (1,T)
        mask = torch.where(kv_idx <= abs_q, 0.0, float("-inf")).float()  # (lt,T)
        rep = n_q // n_kv
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            residual = h
            x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_eps)
            ap = p + "self_attn."
            qkv = self._lin(x, ap + "qkv_proj.weight", w.get(ap + "qkv_proj.bias"))
            q, k, v = qkv.split((n_q * hd, n_kv * hd, n_kv * hd), dim=-1)
            q = q.view(lt, n_q, hd)
            k = k.view(lt, n_kv, hd)
            v = v.view(lt, n_kv, hd)
            q = q * cos_h + _rotate_half(q) * sin_h
            k = k * cos_h + _rotate_half(k) * sin_h
            k_full = cp_all_gather(k).repeat_interleave(rep, dim=1)        # (T,n_q,hd)
            v_full = cp_all_gather(v).repeat_interleave(rep, dim=1)
            qh = q.transpose(0, 1)                                        # (n_q,lt,hd)
            kh = k_full.transpose(0, 1)
            vh = v_full.transpose(0, 1)
            scores = (qh.float() @ kh.float().transpose(-1, -2)) * scale + mask
            attn = torch.softmax(scores, dim=-1).to(self.dtype)
            o = (attn @ vh).transpose(0, 1).reshape(lt, n_q * hd)
            o = o @ w[p + "self_attn.o_proj.weight"].t()
            h = residual + o
            residual = h
            x = _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
            gate, up = self._lin(x, p + "mlp.gate_up_proj.weight").chunk(2, dim=-1)
            h = residual + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].t()
        h = _rms_norm(h, w["model.norm.weight"], cfg.rms_eps)
        return h.float() @ w["lm_head.weight"].float().t()

    def _ffn(self, i, x, prefix, ctx):
        from auto_infer.distributed.parallel_state import tp_all_reduce
        gate_up = self._lin(x, prefix + "mlp.gate_up_proj.weight")
        if self.use_custom_kernels:
            gate, up = gate_up.chunk(2, dim=-1)
            inter = self._swiglu(gate, up)
        else:
            import torch_npu
            inter = torch_npu.npu_swiglu(gate_up)
        return tp_all_reduce(self._lin(inter, prefix + "mlp.down_proj.weight"))
