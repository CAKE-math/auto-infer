"""Fused MoE 对拍 (spec §5/§8): grouped-GEMM fused routed experts == naive per-expert
loop, on real NPU (DeepSeek-V2-Lite, single card, ep=1).
  1. naive greedy reference (per-expert loop)
  2. per-layer numeric 对拍: fused _moe vs naive _moe on the same input (bf16 tol)
  3. fused-only greedy (frees per-expert originals to avoid 2x expert memory) ==
     the naive greedy sequence, token-for-token.
"""
import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device

dev = npu_device(0); path = "/data1/models/DeepSeek-V2-Lite-Chat"
tok = AutoTokenizer.from_pretrained(path)
m = DeepseekV2Model.from_pretrained(path, dev, torch.bfloat16)
cfg = m.cfg
IDS = [tok.bos_token_id] + tok("The capital of France is", add_special_tokens=False).input_ids


def greedy(n=12):
    seq = list(IDS)
    for _ in range(n):
        t = torch.tensor(seq, device=dev); pos = torch.arange(len(seq), device=dev)
        seq.append(int(m.forward_dense(t, pos)[-1].argmax()))
    return seq[len(IDS):]


# 1. naive greedy reference (fused disabled, originals intact)
m.moe.fused = False
naive_seq = greedy()

# 2. per-layer numeric 对拍 at the first MoE layer (naive first, then fused)
i_moe = cfg.first_k_dense
torch.manual_seed(0)
x = torch.randn(7, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
naive_out = m.moe._naive(x, i_moe).float()
fused_out = m.moe._fused_compute(x, i_moe).float()
max_abs = float((naive_out - fused_out).abs().max())
rel = max_abs / (naive_out.abs().max().item() + 1e-6)
close = torch.allclose(naive_out, fused_out, atol=5e-2, rtol=5e-2)
print(f"per-layer MoE 对拍: max|Δ|={max_abs:.4f} rel={rel:.4f} allclose={close}")

# 3. fused whole-model greedy via the DEFAULT dispatch path (fused merged as default)
m.moe.fused = True
fused_seq = greedy()
print(f"naive seq: {naive_seq}")
print(f"fused seq: {fused_seq}")
seq_eq = naive_seq == fused_seq
print("=== FUSED MoE == naive MoE (DeepSeek-V2-Lite, NPU) ===")
print("ALL MATCH" if (close and seq_eq) else "MISMATCH")
