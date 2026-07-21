"""V3 ARCHITECTURE code-path verification on a REAL DeepSeek-V3 config + REAL V3
weight names (yujiepan/deepseek-v3-tiny-random; random weights, so no output
correctness — this checks the FORWARD RUNS end-to-end: MLA q-lora attention
(q_a_proj/q_a_layernorm/q_b_proj + kv_a_proj_with_mqa/kv_a_layernorm/kv_b_proj),
dense layer-0 (first_k_dense_replace=1), MoE layer-1 with V3 sigmoid + noaux_tc
group-limited gating + e_score_correction_bias over 256 experts, YaRN rope —
producing finite logits of the right shape). Verifies my deepseek_v2.py V3
parametrization against the actual V3 checkpoint naming, beyond hand-built configs.
"""
import torch

from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device

dev = npu_device(0); path = "/data2/models/dsv3-tiny-random"
m = DeepseekV2Model.from_pretrained(path, dev, torch.bfloat16)
cfg = m.cfg
print(f"config: q_lora_rank={cfg.q_lora_rank} scoring={cfg.scoring_func} "
      f"topk_method={cfg.topk_method} n_group={cfg.n_group} topk_group={cfg.topk_group} "
      f"n_routed={cfg.n_routed} top_k={cfg.top_k} first_k_dense={cfg.first_k_dense}")
assert cfg.q_lora_rank is not None, "expected V3 q-lora attention"
assert cfg.scoring_func == "sigmoid" and cfg.topk_method == "noaux_tc", "expected V3 gating"

ids = torch.tensor([1, 2, 3, 4, 5, 6], device=dev)
pos = torch.arange(6, device=dev)
logits = m.forward_dense(ids, pos)                      # full V3 forward (MLA q-lora + dense + MoE)
print(f"logits shape={tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} "
      f"dtype={logits.dtype}")
ok = (logits.shape == (6, cfg.vocab_size)) and bool(torch.isfinite(logits).all())
# also exercise the V3 gate directly to confirm group-limited path runs on real gate weights
router = torch.randn(4, cfg.n_routed, device=dev)
w_, idx_ = m.moe._gate(router.float(), "model.layers.1.mlp.")
print(f"V3 gate: selected {tuple(idx_.shape)} experts/token (top_k={cfg.top_k}), "
      f"weights finite={bool(torch.isfinite(w_).all())}")
ok = ok and idx_.shape == (4, cfg.top_k) and bool(torch.isfinite(w_).all())
print("=== V3 architecture forward runs on real V3 config + real weight names ===")
print("ALL PASS" if ok else "FAIL")
