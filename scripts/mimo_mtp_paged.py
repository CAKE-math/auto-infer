"""MTP spec-decode with KV-REUSE (real speedup) on MiMo-7B, single sequence.

Both the target (36-layer, via model.forward) AND the MTP head (1 layer) decode
INCREMENTALLY over paged KV — no dense recompute. Each step:
  1. target forward over [last_confirmed, prev_draft] (2 query positions) reusing
     the main paged KV -> pre-norm hidden + logits;
  2. branchless greedy verify of prev_draft -> emit 1 or 2 tokens;
  3. MTP layer (its OWN 1-layer paged KV) consumes the newly-confirmed position's
     hidden + next token embedding -> next draft.
Plain baseline = the same paged target decoding 1 token/step. Greedy output MUST
match. block_size divides positions so slot_mapping == positions (contiguous).

Honest scope: single sequence (not the batched scheduler) — isolates the
KV-reuse MTP speedup; batched-runner wiring is the productionization.
"""
import json
import sys
import time

import torch  # noqa
import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.layers.attention.gqa import GqaFIABackend
from auto_infer.layers.mlp import swiglu_mlp
from auto_infer.layers.norm import add_rms_norm as _add_rms_norm
from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.models.registry import get_model_class
from auto_infer.platform import npu_device
from auto_infer.forward_context import ForwardContext

path = "/data1/models/MiMo-7B-Base"
dev = npu_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
BS, NBLK = 128, 64                       # block_size | positions -> slot == position
arch = json.load(open(path + "/config.json"))["architectures"][0]
model = get_model_class(arch).from_pretrained(path, device=dev, dtype=torch.bfloat16)
cfg, w = model.cfg, model.w
eps = cfg.rms_eps
embed, head = w["model.embed_tokens.weight"], w["lm_head.weight"]
MP = "model.mtp_layers.0."
BT = torch.arange(NBLK, dtype=torch.int32, device=dev).unsqueeze(0)   # 1 seq, blocks 0..N


def _ctx(be, kv, positions):
    """Paged ForwardContext for one sequence's query at `positions` (a python
    list); kv length = last position + 1. slot == position (contiguous)."""
    T = len(positions)
    pos = torch.tensor(positions, dtype=torch.int64, device=dev)
    ctx = ForwardContext(
        token_ids=None, positions=pos,
        slot_mapping=pos.to(torch.int32),
        block_table=BT, cu_seqlens_q=[T], seqlens_kv=[positions[-1] + 1],
        attn_mask=model_mask, attn_backend=be, kv_caches=kv, is_decode=False)
    return ctx


model_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=dev), diagonal=1)
main_be, main_kv = model.make_attention_backend(NBLK, BS)
mtp_be = GqaFIABackend(n_q_heads=model.n_q_local, n_kv_heads=model.n_kv_local,
                       head_dim=cfg.head_dim, scale=cfg.head_dim ** -0.5, num_layers=1,
                       device=dev, dtype=model.dtype, w=w,
                       layer_prefix=lambda i: MP, rms_eps=eps)
mtp_kv = mtp_be.alloc_kv_caches(NBLK, BS)


@torch.no_grad()
def main_step(token_ids, positions, kv):
    """Run the 36-layer target over query token_ids at positions (writes main KV),
    return (pre_norm_hidden (T,H), logits (T,vocab))."""
    ctx = _ctx(main_be, kv, positions)
    ctx.token_ids = torch.tensor(token_ids, dtype=torch.int64, device=dev)
    cos, sin = model._compute_cos_sin(ctx.positions)
    ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
    h_pre = model.forward(ctx, prenorm=True)
    h_norm = _rms_norm(h_pre, w["model.norm.weight"], eps)
    return h_pre, h_norm.float() @ head.float().t()


@torch.no_grad()
def mtp_forward(prev_hidden, next_tokens, positions):
    """MTP layer over its OWN paged KV for one or more positions i: input =
    combine(hnorm(h_i), enorm(emb(t_{i+1}))) -> input_norm -> paged self-attn
    (writes MTP KV at `positions`) -> mlp -> final_norm -> head. prev_hidden
    (T,H), next_tokens (T,), positions python list. Returns the last position's
    predicted token (the draft for positions[-1]+2)."""
    combined = torch.cat([_rms_norm(prev_hidden, w[MP + "hidden_layernorm.weight"], eps),
                          _rms_norm(embed[next_tokens], w[MP + "token_layernorm.weight"], eps)],
                         -1) @ w[MP + "input_proj.weight"].t()          # (T, H)
    ctx = _ctx(mtp_be, mtp_kv, positions)
    cos, sin = model._compute_cos_sin(ctx.positions)
    ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
    residual = combined
    xn = _rms_norm(combined, w[MP + "input_layernorm.weight"], eps)
    o = mtp_be.attention(0, xn, ctx)
    xa, residual = _add_rms_norm(o, residual, w[MP + "post_attention_layernorm.weight"], eps)
    h = residual + swiglu_mlp(xa, w, MP + "mlp.")
    hn = _rms_norm(h[-1:], w[MP + "final_layernorm.weight"], eps)
    return int((hn.float() @ head.float().t())[-1].argmax())


@torch.no_grad()
def plain(prompt, n):
    for k in main_kv:
        k.zero_()
    seq = list(prompt)
    _, lg = main_step(seq, list(range(len(seq))), main_kv)   # prefill
    t = int(lg[-1].argmax())
    out = [t]
    while len(out) < n:
        _, lg = main_step([t], [len(seq) + len(out) - 1], main_kv)
        t = int(lg[-1].argmax())
        out.append(t)
    return out[:n]


@torch.no_grad()
def spec(prompt, n):
    for k in main_kv:
        k.zero_()
    for k in mtp_kv:
        k.zero_()
    L = len(prompt)
    h_pre, lg = main_step(list(prompt), list(range(L)), main_kv)   # target prefill 0..L-1
    t = int(lg[-1].argmax())                                       # token at pos L
    out = [t]
    # MTP prefill 0..L-2: pos i uses h_i + emb(prompt[i+1]); populates MTP KV so
    # incremental MTP attention has full context. Then MTP at L-1 -> first draft (pos L+1).
    mtp_forward(h_pre[:L - 1], torch.tensor(list(prompt[1:]), device=dev), list(range(L - 1)))
    draft = mtp_forward(h_pre[L - 1:L], torch.tensor([t], device=dev), [L - 1])
    accepted = steps = 0
    while len(out) < n:
        tp = L + len(out) - 1                    # position of last confirmed token t
        h_pre, lg = main_step([t, draft], [tp, tp + 1], main_kv)
        p0 = int(lg[0].argmax())                 # target's real token after t
        steps += 1
        if p0 == draft:                          # accept -> confirm draft + bonus (2 tokens)
            accepted += 1
            bonus = int(lg[1].argmax())          # token after draft (at tp+2)
            mtp_forward(h_pre[0:1], torch.tensor([draft], device=dev), [tp])            # MTP KV @tp
            draft = mtp_forward(h_pre[1:2], torch.tensor([bonus], device=dev), [tp + 1])  # @tp+1 -> next draft
            out += [p0, bonus]                   # p0 == draft (the confirmed token at tp+1)
            t = bonus
        else:                                    # reject -> correction (1 token)
            draft = mtp_forward(h_pre[0:1], torch.tensor([p0], device=dev), [tp])       # MTP KV @tp
            out.append(p0)
            t = p0
    return out[:n], accepted, steps


tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
for name, text in {"code": "def fibonacci(n):\n    if n <= 1:\n        return n\n",
                   "prose": "The history of the Roman Empire began when"}.items():
    prompt = tok(text).input_ids
    N = 96
    t0 = time.time(); base = plain(prompt, N); tb = time.time() - t0
    t0 = time.time(); sp, acc, st = spec(prompt, N); ts = time.time() - t0
    print(f"\n### {name} ###")
    print(f"  greedy MATCH: {base == sp}")
    print(f"  acceptance: {acc}/{st} = {acc / st:.2f}/step -> {1 + acc / st:.2f} tok/step")
    print(f"  plain {tb:.2f}s ({N / tb:.1f} tok/s) | spec {ts:.2f}s ({N / ts:.1f} tok/s)  ->  {tb / ts:.2f}x")
