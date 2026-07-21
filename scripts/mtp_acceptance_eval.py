"""MTP acceptance-rate benchmark over a DATASET (statistically meaningful, vs a
couple hand prompts). Runs paged KV-reuse MTP spec-decode on N real prompts and
aggregates PER-POSITION acceptance a_j = P(draft j hit | reached j) across the
whole set, plus mean tokens/step.

  python scripts/mtp_acceptance_eval.py <device> [dataset] [n_prompts] [gen] [K]
  dataset: gsm8k (default) | mmlu | ceval        greedy output stays exact.
"""
import json
import sys
import time

import torch
import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.layers.attention.gqa import GqaFIABackend
from auto_infer.layers.mlp import swiglu_mlp
from auto_infer.layers.norm import add_rms_norm as _add_rms_norm
from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.models.registry import get_model_class
from auto_infer.platform import npu_device
from auto_infer.spec_decode.rejection_sampler import verify_and_accept
from auto_infer.forward_context import ForwardContext

dev = npu_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
DATASET = sys.argv[2] if len(sys.argv) > 2 else "gsm8k"
N_PROMPTS = int(sys.argv[3]) if len(sys.argv) > 3 else 50
GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 64
K = int(sys.argv[5]) if len(sys.argv) > 5 else 1
path = "/data1/models/MiMo-7B-Base"
BS, NBLK = 128, 96

arch = json.load(open(path + "/config.json"))["architectures"][0]
model = get_model_class(arch).from_pretrained(path, device=dev, dtype=torch.bfloat16)
cfg, w = model.cfg, model.w
eps = cfg.rms_eps
embed, head = w["model.embed_tokens.weight"], w["lm_head.weight"]
MP = "model.mtp_layers.0."
mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=dev), diagonal=1)
BT = torch.arange(NBLK, dtype=torch.int32, device=dev).unsqueeze(0)
main_be, main_kv = model.make_attention_backend(NBLK, BS)
mtp_be = GqaFIABackend(n_q_heads=model.n_q_local, n_kv_heads=model.n_kv_local,
                       head_dim=cfg.head_dim, scale=cfg.head_dim ** -0.5, num_layers=1,
                       device=dev, dtype=model.dtype, w=w, layer_prefix=lambda i: MP, rms_eps=eps)
mtp_kv = mtp_be.alloc_kv_caches(NBLK, BS)


def _ctx(be, kv, positions):
    pos = torch.tensor(positions, dtype=torch.int64, device=dev)
    return ForwardContext(token_ids=None, positions=pos, slot_mapping=pos.to(torch.int32),
                          block_table=BT, cu_seqlens_q=[len(positions)],
                          seqlens_kv=[positions[-1] + 1], attn_mask=mask,
                          attn_backend=be, kv_caches=kv, is_decode=False)


@torch.no_grad()
def main_step(token_ids, positions):
    ctx = _ctx(main_be, main_kv, positions)
    ctx.token_ids = torch.tensor(token_ids, dtype=torch.int64, device=dev)
    cos, sin = model._compute_cos_sin(ctx.positions)
    ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
    h = model.forward(ctx, prenorm=True)
    return h, (_rms_norm(h, w["model.norm.weight"], eps).float() @ head.float().t())


@torch.no_grad()
def mtp_fwd(prev_hidden, next_tokens, positions):
    comb = torch.cat([_rms_norm(prev_hidden, w[MP + "hidden_layernorm.weight"], eps),
                      _rms_norm(embed[next_tokens], w[MP + "token_layernorm.weight"], eps)],
                     -1) @ w[MP + "input_proj.weight"].t()
    ctx = _ctx(mtp_be, mtp_kv, positions)
    cos, sin = model._compute_cos_sin(ctx.positions)
    ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
    res = comb
    xn = _rms_norm(comb, w[MP + "input_layernorm.weight"], eps)
    o = mtp_be.attention(0, xn, ctx)
    xa, res = _add_rms_norm(o, res, w[MP + "post_attention_layernorm.weight"], eps)
    h = res + swiglu_mlp(xa, w, MP + "mlp.")
    hn = _rms_norm(h[-1:], w[MP + "final_layernorm.weight"], eps)
    return hn, int((hn.float() @ head.float().t()).argmax())


def unroll(prev_h, tok, pos, k):
    out = []
    for _ in range(k):
        prev_h, d = mtp_fwd(prev_h, torch.tensor([tok], device=dev), [pos])
        out.append(d); tok = d; pos += 1
    return out


def reset():
    for c in main_kv + mtp_kv:
        c.zero_()


@torch.no_grad()
def spec(prompt, n, reached, acc_pos):
    """Run MTP spec-decode; accumulate per-position reached/accepted into the
    passed-in aggregators. Returns (num_emitted, num_steps)."""
    reset()
    L0 = len(prompt)
    if L0 < 2 or L0 + n >= NBLK * BS:
        return 0, 0
    h_pre, lg = main_step(list(prompt), list(range(L0)))
    t = int(lg[-1].argmax())
    seq = list(prompt) + [t]
    mtp_fwd(h_pre[:L0 - 1], torch.tensor(list(prompt[1:]), device=dev), list(range(L0 - 1)))
    drafts = unroll(h_pre[L0 - 1:L0], t, L0 - 1, K)
    steps = 0
    while len(seq) - L0 < n:
        tp = len(seq) - 1
        pos = [tp] + [tp + 1 + j for j in range(K)]
        h_pre, lg = main_step([t] + drafts, pos)
        preds = lg.argmax(-1)
        m, _, _ = verify_and_accept(torch.tensor([drafts], device=dev), preds.unsqueeze(0))
        m = int(m[0])
        for x in preds[:m + 1].tolist():
            seq.append(x)
            if len(seq) - L0 >= n:
                break
        steps += 1
        for j in range(K):
            if m >= j:
                reached[j] += 1
            if m >= j + 1:
                acc_pos[j] += 1
        last_h, d0 = mtp_fwd(h_pre[:m + 1], preds[:m + 1], [tp + j for j in range(m + 1)])
        drafts = [d0] + unroll(last_h, d0, tp + m + 1, K - 1)
        t = seq[-1]
    return len(seq) - L0, steps


def load_prompts():
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    if DATASET == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [d["question"] for d in ds.select(range(N_PROMPTS))]
    if DATASET == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        return [d["question"] for d in ds.select(range(N_PROMPTS))]
    if DATASET == "ceval":
        ds = load_dataset("ceval/ceval-exam", "default", revision="refs/convert/parquet",
                          split="validation")
        return [d["question"] for d in ds.select(range(N_PROMPTS))]
    raise SystemExit(f"unknown dataset {DATASET}")


tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
texts = load_prompts()
reached, acc_pos = [0] * K, [0] * K
tot_emit = tot_steps = 0
t0 = time.time()
for i, text in enumerate(texts):
    ids = tok(text).input_ids[:400]
    e, s = spec(ids, GEN, reached, acc_pos)
    tot_emit += e; tot_steps += s
elapsed = time.time() - t0
alpha = [acc_pos[j] / reached[j] if reached[j] else 0.0 for j in range(K)]
print(f"\n=== MTP acceptance over {DATASET} ({len(texts)} prompts x {GEN} tok, K={K}) ===")
for j in range(K):
    print(f"  a{j} = {alpha[j]:.3f}   (accepted {acc_pos[j]} / reached {reached[j]})")
print(f"  mean tokens/step = {tot_emit / tot_steps:.3f}   ({tot_emit} tok / {tot_steps} steps)")
print(f"  throughput = {tot_emit / elapsed:.1f} tok/s")
