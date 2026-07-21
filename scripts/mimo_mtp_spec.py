"""MTP speculative decoding on MiMo-7B (Xiaomi) — a REAL trained MTP head.

MiMo base = Qwen2 dense GQA (loads via Qwen2Model); its `model.mtp_layers.0.*` is
a trained MTP module (num_nextn_predict_layers=1): token_layernorm(=enorm),
hidden_layernorm(=hnorm), input_proj(=eh_proj, 2H->H), a full Qwen2 decoder block
(own weights), final_layernorm(=head norm); output uses the model's lm_head.

MTP recurrence (DeepSeek-V3 / MiMo): predict t_{i+2} from hidden h_i and
Emb(t_{i+1}). So with 1 MTP layer we draft ONE token per step; the target then
verifies it (greedy branchless accept) — greedy output stays token-identical to
plain decode, and every accepted draft is a free extra token (up to 2x/step).
"""
import json
import sys
import time

import torch  # noqa
import torch_npu  # noqa
from transformers import AutoTokenizer

from auto_infer.layers.norm import rms_norm as _rms_norm
from auto_infer.models.registry import get_model_class
from auto_infer.platform import npu_device

path = "/data1/models/MiMo-7B-Base"
dev = npu_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
arch = json.load(open(path + "/config.json"))["architectures"][0]
model = get_model_class(arch).from_pretrained(path, device=dev, dtype=torch.bfloat16)
cfg, w = model.cfg, model.w
H, eps = cfg.hidden_size, cfg.rms_eps
embed, head = w["model.embed_tokens.weight"], w["lm_head.weight"]

# The MTP decoder block is a Qwen2 layer with its OWN weights — alias them to a
# synthetic layer index so base.run_layer_dense (GQA dense) can execute it.
MP, LAYER = "model.mtp_layers.0.", cfg.num_layers
for suf in ["input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight"]:
    w[f"model.layers.{LAYER}." + suf] = w[MP + suf]
enorm, hnorm = w[MP + "token_layernorm.weight"], w[MP + "hidden_layernorm.weight"]
eh_proj, fnorm = w[MP + "input_proj.weight"], w[MP + "final_layernorm.weight"]


@torch.no_grad()
def mtp_draft_one(seq, t_next):
    """Draft t_{L+1} = MTP( h_{L-1}, Emb(t_L=t_next) ). Runs the MTP block over the
    full context (positions 0..L-1) pairing each hidden with the NEXT token's
    embedding (shifted), so its self-attn sees context; take the last position."""
    ids = torch.tensor(seq, device=dev)
    pos = torch.arange(len(seq), device=dev)
    h = model.hidden_dense(ids, pos, prenorm=True)                    # (L, H) pre-final-norm hidden (MTP input)
    shifted = torch.tensor(seq[1:] + [t_next], device=dev)            # emb of the NEXT token
    combined = torch.cat([_rms_norm(h, hnorm, eps),
                          _rms_norm(embed[shifted], enorm, eps)], -1) @ eh_proj.t()   # (L, H)
    hm = model.run_layer_dense(combined, pos, LAYER)                  # (L, H) MTP block
    return int((_rms_norm(hm[-1:], fnorm, eps).float() @ head.float().t()).argmax())


@torch.no_grad()
def greedy_next(ids):
    return model.forward_dense(torch.tensor(ids, device=dev),
                               torch.arange(len(ids), device=dev))


@torch.no_grad()
def plain(prompt, n):
    seq = list(prompt)
    for _ in range(n):
        seq.append(int(greedy_next(seq)[-1].argmax()))
    return seq[len(prompt):]


@torch.no_grad()
def spec(prompt, n):
    seq = list(prompt)
    t_next = int(greedy_next(seq)[-1].argmax())          # first definite token from the target
    accepted = steps = 0
    while len(seq) - len(prompt) < n:
        d = mtp_draft_one(seq, t_next)                    # 1 MTP draft (t after t_next)
        vlog = greedy_next(seq + [t_next, d])
        p1 = int(vlog[len(seq)].argmax())                 # target's real token after t_next
        steps += 1
        seq.append(t_next)                                # t_next always confirmed
        if p1 == d and len(seq) - len(prompt) < n:
            accepted += 1
            seq.append(d)                                 # draft matched -> free token
            t_next = int(vlog[len(seq) - 1].argmax())     # token after d
        else:
            t_next = p1                                   # correction
    return seq[len(prompt):len(prompt) + n], accepted, steps


tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
for name, text in {"code": "def fibonacci(n):\n    if n <= 1:\n        return n\n",
                   "prose": "The history of the Roman Empire began when"}.items():
    prompt = tok(text).input_ids
    N = 64
    t0 = time.time(); base = plain(prompt, N); tb = time.time() - t0
    t0 = time.time(); sp, acc, steps = spec(prompt, N); ts = time.time() - t0
    print(f"\n### {name} ###")
    print(f"  greedy MATCH: {base == sp}")
    print(f"  acceptance: {acc}/{steps} drafts = {acc / steps:.2f}/step -> "
          f"{1 + acc / steps:.2f} tokens/step")
    print(f"  plain {tb:.1f}s | spec {ts:.1f}s ({tb / ts:.2f}x)  [note: dense recompute, not KV-opt]")
