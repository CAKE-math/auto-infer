"""RoPE variants. Default = NeoX-style rotate_half (Qwen/Llama); DeepSeek V2/V3
add interleaved-layout + YaRN scaling. This is the single home for rope math —
models pass their config in and read back inv_freq/scales, rather than each
model reimplementing YaRN inline."""
import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def ds_rope_interleave(x):
    """DeepSeek-V2 RoPE layout fix: the pe dims are stored INTERLEAVED in the
    checkpoint, so HF's `apply_rotary_pos_emb` reshapes q/k via
    `view(..., d//2, 2).transpose(-1,-2).reshape(..., d)` BEFORE the standard
    half-split rotation. Without this the positional encoding is wrong every
    layer and compounds into incoherent output (was the coherent-generation bug
    the HF parity bisect localized)."""
    d = x.shape[-1]
    return x.view(*x.shape[:-1], d // 2, 2).transpose(-1, -2).reshape(*x.shape[:-1], d)


# --- YaRN rope scaling (DeepSeek V2/V3) ---------------------------------------
def _yarn_mscale(scale, mscale=1.0):
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_correction_dim(num_rot, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(_yarn_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_ramp(low, high, dim, device):
    if low == high:
        high += 0.001
    lin = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return lin.clamp(0, 1)


def build_rope_inv_freq(dim, base, rope_scaling, device):
    """Return (inv_freq, cos_sin_mscale, softmax_scale_mult) for a rope head dim.

    Plain NeoX when `rope_scaling` is None/not-yarn (mscale=1, softmax mult=1).
    For YaRN (DeepSeek V2/V3) it interpolates extrapolation/interpolation freqs
    over the correction range and returns the attention-entropy `mscale` and the
    softmax-scale multiplier the model folds into its own softmax_scale."""
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freq_extra = 1.0 / (base ** (idx / dim))
    rs = rope_scaling
    if rs and rs.get("type") == "yarn":
        factor = rs["factor"]
        orig = rs["original_max_position_embeddings"]
        mscale = rs.get("mscale", 1.0)
        mscale_all = rs.get("mscale_all_dim", 0.0)
        freq_inter = 1.0 / (factor * base ** (idx / dim))
        low, high = _yarn_correction_range(rs["beta_fast"], rs["beta_slow"], dim, base, orig)
        mask = 1.0 - _yarn_ramp(low, high, dim // 2, device)
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        cs_mscale = _yarn_mscale(factor, mscale) / _yarn_mscale(factor, mscale_all)
        ss_mult = _yarn_mscale(factor, mscale_all) ** 2
        return inv_freq, cs_mscale, ss_mult
    return freq_extra, 1.0, 1.0
