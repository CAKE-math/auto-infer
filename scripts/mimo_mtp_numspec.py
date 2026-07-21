"""num_spec>1 experiment: draft K tokens from MiMo's SINGLE MTP layer by
recurrent unroll (EAGLE-style: feed the MTP's own hidden + its own draft back).
Single sequence, paged KV-reuse. Sweeps K=1..4 and reports acceptance / tokens-
per-step / speedup — expect diminishing returns (one layer -> later drafts poor).
Greedy output stays token-identical to plain (verify emits the target argmax)."""
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
path = "/data1/models/MiMo-7B-Base"
BS, NBLK = 128, 64
import json
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
    h_pre = model.forward(ctx, prenorm=True)
    return h_pre, (_rms_norm(h_pre, w["model.norm.weight"], eps).float() @ head.float().t())


@torch.no_grad()
def mtp_fwd(prev_hidden, next_tokens, positions):
    """MTP layer over its paged KV; returns (last-position pre-final-norm hidden
    (1,H), last-position draft token). Writes MTP KV at `positions`."""
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
    hn = _rms_norm(h[-1:], w[MP + "final_layernorm.weight"], eps)     # post-final-norm
    d = int((hn.float() @ head.float().t()).argmax())
    # recurrence feeds the MTP's POST-final-norm output (matches vLLM MiMoMTP,
    # whose forward RETURNS final_layernorm(...) and feeds it back as prev_hidden)
    return hn, d


def unroll(prev_h, tok, pos, K):
    """K drafts: 1st from (prev_h, tok) at `pos`, then recurrently feed the MTP's
    own hidden + draft (single layer -> proxy for the main hidden)."""
    drafts = []
    for _ in range(K):
        prev_h, d = mtp_fwd(prev_h, torch.tensor([tok], device=dev), [pos])
        drafts.append(d); tok = d; pos += 1
    return drafts


def reset():
    for c in main_kv + mtp_kv:
        c.zero_()


@torch.no_grad()
def plain(prompt, n):
    reset()
    _, lg = main_step(list(prompt), list(range(len(prompt))))
    seq = list(prompt) + [int(lg[-1].argmax())]
    while len(seq) - len(prompt) < n:
        h, lg = main_step([seq[-1]], [len(seq) - 1])
        seq.append(int(lg[-1].argmax()))
    return seq[len(prompt):n + len(prompt)]


@torch.no_grad()
def spec(prompt, n, K):
    reset()
    L0 = len(prompt)
    h_pre, lg = main_step(list(prompt), list(range(L0)))
    t = int(lg[-1].argmax())
    seq = list(prompt) + [t]
    # MTP prefill 0..L0-2, then K drafts unrolled from position L0-1
    mtp_fwd(h_pre[:L0 - 1], torch.tensor(list(prompt[1:]), device=dev), list(range(L0 - 1)))
    drafts = unroll(h_pre[L0 - 1:L0], t, L0 - 1, K)
    acc = steps = 0
    reached = [0] * K          # draft j was CHECKED (drafts 0..j-1 all accepted)
    acc_pos = [0] * K          # draft j MATCHED
    while len(seq) - len(prompt) < n:
        tp = len(seq) - 1                                  # position of last confirmed token t
        pos = [tp] + [tp + 1 + j for j in range(K)]
        h_pre, lg = main_step([t] + drafts, pos)
        preds = lg.argmax(-1)                              # (K+1,) p0..pK
        m, _, _ = verify_and_accept(torch.tensor([drafts], device=dev), preds.unsqueeze(0))
        m = int(m[0])
        emit = preds[:m + 1].tolist()                      # p0..pm
        for x in emit:
            seq.append(x)
            if len(seq) - len(prompt) >= n:
                break
        acc += m; steps += 1
        for j in range(K):                                 # per-position acceptance
            if m >= j:
                reached[j] += 1
            if m >= j + 1:
                acc_pos[j] += 1
        # advance MTP over the m+1 confirmed positions (real hidden) -> d0, then unroll
        last_h, d0 = mtp_fwd(h_pre[:m + 1], torch.tensor(emit, device=dev),
                             [tp + j for j in range(m + 1)])
        drafts = [d0] + unroll(last_h, d0, tp + m + 1, K - 1)
        t = seq[-1]
    alpha = [acc_pos[j] / reached[j] if reached[j] else 0.0 for j in range(K)]
    return seq[len(prompt):n + len(prompt)], acc, steps, alpha, reached


tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
TEXTS = {"code": "def fibonacci(n):\n    if n <= 1:\n        return n\n",
         "prose": "The history of the Roman Empire began when"}
N = 96
for name, text in TEXTS.items():
    ids = tok(text).input_ids
    t0 = time.time(); base = plain(ids, N); tb = time.time() - t0
    print(f"\n### {name} (plain {N / tb:.1f} tok/s) ###")
    for K in (1, 2, 3, 4):
        t0 = time.time(); sp, acc, st, alpha, reached = spec(ids, N, K); ts = time.time() - t0
        apos = " ".join(f"a{j}={a:.2f}(n={reached[j]})" for j, a in enumerate(alpha))
        print(f"  K={K}: MATCH={base == sp}  [{apos}]  {1 + acc / st:.2f} tok/step  {tb / ts:.2f}x")
