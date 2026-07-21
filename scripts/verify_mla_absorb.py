"""MLA absorption math 对拍 (spec §6 core DeepSeek optimization): the absorbed form
(cache only latent ckv[kv_lora]+k_pe[rope], project query into latent via W_UK, absorb
W_UV into output) == the non-absorbed form (materialize full K=nope+rope, V=vd per head).
Uses real DeepSeek-V2-Lite layer weights + one full-attention step (prefill). If the
einsum absorption is correct, outputs match within bf16 tol. Proves the math before
wiring the paged latent-cache (which shrinks KV ~9x)."""
import torch
import torch_npu  # noqa: F401
from auto_infer.models.deepseek_v2 import DeepseekV2Model, _rms_norm, _rotate_half
from auto_infer.platform import npu_device

dev = npu_device(0); path = "/data1/models/DeepSeek-V2-Lite-Chat"
m = DeepseekV2Model.from_pretrained(path, dev, torch.bfloat16)
cfg = m.cfg
nh, nope, rope, vd, kvl = cfg.num_heads, cfg.qk_nope, cfg.qk_rope, cfg.v_head_dim, cfg.kv_lora_rank
prefix = "model.layers.3."          # a MoE layer's attention (attention identical across layers)
w = m.w
T = 6
torch.manual_seed(0)
x = torch.randn(T, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
pos = torch.arange(T, device=dev)
cos, sin = m._cos_sin(pos); cos1, sin1 = cos.unsqueeze(1), sin.unsqueeze(1)
scale = m._softmax_scale
causal = torch.triu(torch.full((T, T), float("-inf"), device=dev, dtype=torch.float32), 1).unsqueeze(1)  # (T,1,T) broadcast over heads

# shared projections
if cfg.q_lora_rank is None:
    q = x @ w[prefix + "self_attn.q_proj.weight"].t()
else:
    q = _rms_norm(x @ w[prefix + "self_attn.q_a_proj.weight"].t(), w[prefix + "self_attn.q_a_layernorm.weight"], cfg.rms_eps)
    q = q @ w[prefix + "self_attn.q_b_proj.weight"].t()
q = q.view(T, nh, nope + rope)
q_nope, q_pe = q.split([nope, rope], dim=-1)
ckv_full = x @ w[prefix + "self_attn.kv_a_proj_with_mqa.weight"].t()
ckv, k_pe = ckv_full.split([kvl, rope], dim=-1)
ckv = _rms_norm(ckv, w[prefix + "self_attn.kv_a_layernorm.weight"], cfg.rms_eps)      # (T, kvl)
k_pe = k_pe.view(T, 1, rope)
q_pe = q_pe * cos1 + _rotate_half(q_pe) * sin1
k_pe = k_pe * cos1 + _rotate_half(k_pe) * sin1                                    # (T,1,rope)

# --- non-absorbed reference ---
kv = (ckv @ w[prefix + "self_attn.kv_b_proj.weight"].t()).view(T, nh, nope + vd)
k_nope, v = kv.split([nope, vd], dim=-1)
k = torch.cat([k_nope, k_pe.expand(T, nh, rope)], dim=-1)
qf = torch.cat([q_nope, q_pe], dim=-1)
scores = torch.einsum("tnd,snd->tns", qf.float(), k.float()) * scale + causal
attn = torch.softmax(scores, dim=-1).to(x.dtype)
o_ref = torch.einsum("tns,snv->tnv", attn.float(), v.float())                     # (T,nh,vd)

# --- absorbed form ---
W = w[prefix + "self_attn.kv_b_proj.weight"].view(nh, nope + vd, kvl)                 # (nh, nope+vd, kvl)
W_UK, W_UV = W.split([nope, vd], dim=1)                                            # (nh,nope,kvl),(nh,vd,kvl)
q_absorbed = torch.einsum("tnp,npl->tnl", q_nope.float(), W_UK.float())           # (T,nh,kvl)
score_nope = torch.einsum("tnl,sl->tns", q_absorbed, ckv.float())
score_pe = torch.einsum("tnr,sr->tns", q_pe.float(), k_pe.squeeze(1).float())
scores_a = (score_nope + score_pe) * scale + causal
attn_a = torch.softmax(scores_a, dim=-1)
o_latent = torch.einsum("tns,sl->tnl", attn_a, ckv.float())                        # (T,nh,kvl)
o_abs = torch.einsum("tnl,nvl->tnv", o_latent, W_UV.float())                       # (T,nh,vd)

d = float((o_ref - o_abs).abs().max()); rel = d / (o_ref.abs().max().item() + 1e-6)
close = torch.allclose(o_ref, o_abs, atol=2e-2, rtol=2e-2)
print(f"MLA absorb 对拍: max|Δ|={d:.4f} rel={rel:.4f} allclose={close}")
kv_nonabs = nh * (nope + rope) + nh * vd
kv_abs = kvl + rope
print(f"KV/token/layer: non-absorbed={kv_nonabs} absorbed={kv_abs} ({kv_nonabs/kv_abs:.1f}x smaller)")
print("=== MLA absorption math == non-absorbed ===")
print("ALL MATCH" if close else "MISMATCH")
