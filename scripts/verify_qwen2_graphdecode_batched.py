"""Step 3a: BATCHED gear-bucketed ACL graph decode. Two different prompts decoded
together inside one gear-4 graph (B=2, padded to 4); each must match its own
independent eager decode. Proves batched per-seq actual_seq_kvlen update +
per-seq block_table rows + padding under one captured graph.
TND rule: actual_seq_qlen is CUMULATIVE; actual_seq_kvlen is PER-SEQUENCE.
"""
import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.qwen2 import Qwen2Model, _rms_norm, _rotate_half
from auto_infer.platform import npu_device

dev = npu_device(0); path = "/data0/models/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
m = Qwen2Model.from_pretrained(path, dev, torch.bfloat16)
cfg = m.cfg; w = m.w; NZ = 16
nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim; scale = hd ** -0.5
nb, bs = 64, 128
GEAR = 4
QLEN_CUM = list(range(1, GEAR + 1))       # cumulative q offsets: [1,2,3,4]
mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=dev))
kcs = [torch.zeros(nb, bs, nkv, hd, dtype=torch.bfloat16, device=dev) for _ in range(cfg.num_layers)]
vcs = [torch.zeros(nb, bs, nkv, hd, dtype=torch.bfloat16, device=dev) for _ in range(cfg.num_layers)]
bt = torch.tensor([[r] for r in range(GEAR)], dtype=torch.int32, device=dev)   # seq r -> block r


def zero_cache():
    for c in kcs: c.zero_()
    for c in vcs: c.zero_()


def store(i, k, v, slot):
    torch_npu.npu_scatter_pa_kv_cache(
        k.contiguous(), v.contiguous(),
        kcs[i].view(nb, nkv * hd // NZ, bs, NZ), vcs[i].view(nb, nkv * hd // NZ, bs, NZ), slot)


def prefill_seq(ids, seqr):
    P = len(ids); h = w["model.embed_tokens.weight"][torch.tensor(ids, device=dev)]
    pos = torch.arange(P, device=dev)
    slot = torch.arange(P, dtype=torch.int32, device=dev) + seqr * bs
    cos, sin = m._rope_cos_sin(pos); c = cos.unsqueeze(1); s = sin.unsqueeze(1)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."; res = h; x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_eps)
        q = (x @ w[p + "self_attn.q_proj.weight"].t() + w[p + "self_attn.q_proj.bias"]).view(P, nh, hd)
        k = (x @ w[p + "self_attn.k_proj.weight"].t() + w[p + "self_attn.k_proj.bias"]).view(P, nkv, hd)
        v = (x @ w[p + "self_attn.v_proj.weight"].t() + w[p + "self_attn.v_proj.bias"]).view(P, nkv, hd)
        q = q * c + _rotate_half(q) * s; k = k * c + _rotate_half(k) * s; store(i, k, v, slot)
        o = torch_npu.npu_fused_infer_attention_score_v2(
            q, k, v, num_query_heads=nh, num_key_value_heads=nkv, input_layout="TND",
            softmax_scale=scale, sparse_mode=3, atten_mask=mask,
            actual_seq_qlen=[P], actual_seq_kvlen=[P], next_tokens=0)[0].view(P, nh, hd)
        o = o.reshape(P, nh * hd) @ w[p + "self_attn.o_proj.weight"].t(); h = res + o
        res = h; x = _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
        g = x @ w[p + "mlp.gate_proj.weight"].t(); u = x @ w[p + "mlp.up_proj.weight"].t()
        h = res + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].t()
    hn = _rms_norm(h, w["model.norm.weight"], cfg.rms_eps)
    return int((hn[-1].float() @ w["lm_head.weight"].float().t()).argmax())


tid = torch.zeros(GEAR, dtype=torch.long, device=dev)
ppos = torch.zeros(GEAR, dtype=torch.long, device=dev)
pslot = torch.zeros(GEAR, dtype=torch.int32, device=dev)
hout = torch.zeros(GEAR, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
HANDLES = []; PARAMS = []


def decode_fwd(capturing, kvlens=None):
    if kvlens is None: kvlens = [1] * GEAR
    h = w["model.embed_tokens.weight"][tid]
    cos, sin = m._rope_cos_sin(ppos); c = cos.unsqueeze(1); s = sin.unsqueeze(1)
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."; res = h; x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_eps)
        q = (x @ w[p + "self_attn.q_proj.weight"].t() + w[p + "self_attn.q_proj.bias"]).view(GEAR, nh, hd)
        k = (x @ w[p + "self_attn.k_proj.weight"].t() + w[p + "self_attn.k_proj.bias"]).view(GEAR, nkv, hd)
        v = (x @ w[p + "self_attn.v_proj.weight"].t() + w[p + "self_attn.v_proj.bias"]).view(GEAR, nkv, hd)
        q = q * c + _rotate_half(q) * s; k = k * c + _rotate_half(k) * s; store(i, k, v, pslot)
        knz = kcs[i].view(-1, nkv, hd // NZ, bs, NZ); vnz = vcs[i].view(-1, nkv, hd // NZ, bs, NZ)
        o = torch.empty(GEAR, nh, hd, dtype=h.dtype, device=dev)
        lse = torch.empty(GEAR, dtype=torch.float32, device=dev)
        if capturing:
            st = torch.npu.current_stream(); torch.npu.graph_task_group_begin(st)
            torch_npu.npu_fused_infer_attention_score_v2.out(
                q, knz, vnz, num_query_heads=nh, num_key_value_heads=nkv, input_layout="TND",
                softmax_scale=scale, block_table=bt, block_size=bs, sparse_mode=3, atten_mask=mask,
                actual_seq_qlen=QLEN_CUM, actual_seq_kvlen=kvlens, out=[o, lse])
            hd_ = torch.npu.graph_task_group_end(st); HANDLES.append(hd_); PARAMS.append((q, knz, vnz, o, lse))
        else:
            torch_npu.npu_fused_infer_attention_score_v2.out(
                q, knz, vnz, num_query_heads=nh, num_key_value_heads=nkv, input_layout="TND",
                softmax_scale=scale, block_table=bt, block_size=bs, sparse_mode=3, atten_mask=mask,
                actual_seq_qlen=QLEN_CUM, actual_seq_kvlen=kvlens, out=[o, lse])
        o = o.reshape(GEAR, nh * hd) @ w[p + "self_attn.o_proj.weight"].t(); h = res + o
        res = h; x = _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
        g = x @ w[p + "mlp.gate_proj.weight"].t(); u = x @ w[p + "mlp.up_proj.weight"].t()
        h = res + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].t()
    hout.copy_(_rms_norm(h, w["model.norm.weight"], cfg.rms_eps))


def upd(kvlens):
    st = torch.npu.current_stream()
    for hdl, (q, knz, vnz, o, lse) in zip(HANDLES, PARAMS):
        torch.npu.graph_task_update_begin(st, hdl)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            q, knz, vnz, num_query_heads=nh, num_key_value_heads=nkv, input_layout="TND",
            softmax_scale=scale, block_table=bt, block_size=bs, sparse_mode=3, atten_mask=mask,
            actual_seq_qlen=QLEN_CUM, actual_seq_kvlen=kvlens, out=[o, lse])
        torch.npu.graph_task_update_end(st)


def tok_at(r):
    return int((hout[r].float() @ w["lm_head.weight"].float().t()).argmax())


def eager_ref(ids, n):
    """Clean single-seq eager decode in block 0; row0 carries the seq, kvlen row0 real."""
    zero_cache(); first = prefill_seq(ids, 0); seq = list(ids) + [first]
    for _ in range(n):
        pp = len(seq) - 1
        for r in range(GEAR):
            tid[r] = seq[-1]; ppos[r] = pp; pslot[r] = r * bs + pp
        decode_fwd(False, kvlens=[pp + 1] * GEAR)
        seq.append(tok_at(0))
    return seq


idsA = tok("The capital of France is").input_ids
idsB = tok("Water is made of hydrogen and").input_ids
refA = eager_ref(idsA, 10); refB = eager_ref(idsB, 10)
print("REF A:", repr(tok.decode(refA)))
print("REF B:", repr(tok.decode(refB)))

# batched graph decode: A->block0, B->block1, rows 2,3 padding
zero_cache(); fA = prefill_seq(idsA, 0); fB = prefill_seq(idsB, 1)
seqA = list(idsA) + [fA]; seqB = list(idsB) + [fB]
# warmup+capture into far scratch blocks (10..13) so store() never touches real blocks 0..3
SAFE = torch.tensor([(10 + r) * bs for r in range(GEAR)], dtype=torch.int32, device=dev)
tid.fill_(0); ppos.fill_(0); pslot.copy_(SAFE); decode_fwd(False)   # warmup (scratch slots)
g = torch.npu.NPUGraph()
with torch.npu.graph(g): decode_fwd(True)
for _ in range(10):
    ppA = len(seqA) - 1; ppB = len(seqB) - 1
    tid[0] = seqA[-1]; ppos[0] = ppA; pslot[0] = 0 * bs + ppA
    tid[1] = seqB[-1]; ppos[1] = ppB; pslot[1] = 1 * bs + ppB
    tid[2] = 0; ppos[2] = 0; pslot[2] = 2 * bs
    tid[3] = 0; ppos[3] = 0; pslot[3] = 3 * bs
    upd([ppA + 1, ppB + 1, 1, 1])
    g.replay(); torch.npu.synchronize()
    seqA.append(tok_at(0)); seqB.append(tok_at(1))
print("GPH A:", repr(tok.decode(seqA)))
print("GPH B:", repr(tok.decode(seqB)))
okA = seqA == refA; okB = seqB == refB
print(f"MATCH A={okA} B={okB} -> batched gear graph-decode {'OK' if okA and okB else 'MISMATCH'}")
