"""Step 2: capture whole-model B=1 decode into ACL graph (per-layer graph_task_group),
replay each step with graph_task_update(context_lens). Verify == eager Paris."""
import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.qwen2 import Qwen2Model, _rms_norm, _rotate_half
from auto_infer.platform import npu_device
dev=npu_device(0); path="/data0/models/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(path); m=Qwen2Model.from_pretrained(path,dev,torch.bfloat16)
cfg=m.cfg; w=m.w; NZ=16; nh,nkv,hd=cfg.num_heads,cfg.num_kv_heads,cfg.head_dim; scale=hd**-0.5
nb,bs=64,128
mask=~torch.tril(torch.ones((2048,2048),dtype=torch.bool,device=dev))
kcs=[torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev) for _ in range(cfg.num_layers)]
vcs=[torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev) for _ in range(cfg.num_layers)]
bt=torch.tensor([[0]],dtype=torch.int32,device=dev)
def store(i,k,v,slot):
    torch_npu.npu_scatter_pa_kv_cache(k.contiguous(),v.contiguous(),
        kcs[i].view(nb,nkv*hd//NZ,bs,NZ),vcs[i].view(nb,nkv*hd//NZ,bs,NZ),slot)
def prefill(ids):  # eager prefill to populate cache
    P=len(ids); h=w["model.embed_tokens.weight"][torch.tensor(ids,device=dev)]
    pos=torch.arange(P,device=dev); slot=torch.arange(P,dtype=torch.int32,device=dev)
    cos,sin=m._rope_cos_sin(pos); c=cos.unsqueeze(1); s=sin.unsqueeze(1)
    for i in range(cfg.num_layers):
        p=f"model.layers.{i}."; res=h; x=_rms_norm(h,w[p+"input_layernorm.weight"],cfg.rms_eps)
        q=(x@w[p+"self_attn.q_proj.weight"].t()+w[p+"self_attn.q_proj.bias"]).view(P,nh,hd)
        k=(x@w[p+"self_attn.k_proj.weight"].t()+w[p+"self_attn.k_proj.bias"]).view(P,nkv,hd)
        v=(x@w[p+"self_attn.v_proj.weight"].t()+w[p+"self_attn.v_proj.bias"]).view(P,nkv,hd)
        q=q*c+_rotate_half(q)*s; k=k*c+_rotate_half(k)*s; store(i,k,v,slot)
        o=torch_npu.npu_fused_infer_attention_score_v2(q,k,v,num_query_heads=nh,num_key_value_heads=nkv,
            input_layout="TND",softmax_scale=scale,sparse_mode=3,atten_mask=mask,actual_seq_qlen=[P],actual_seq_kvlen=[P],next_tokens=0)[0].view(P,nh,hd)
        o=o.reshape(P,nh*hd)@w[p+"self_attn.o_proj.weight"].t(); h=res+o
        res=h; x=_rms_norm(h,w[p+"post_attention_layernorm.weight"],cfg.rms_eps)
        g=x@w[p+"mlp.gate_proj.weight"].t(); u=x@w[p+"mlp.up_proj.weight"].t(); h=res+(torch.nn.functional.silu(g)*u)@w[p+"mlp.down_proj.weight"].t()
    hn=_rms_norm(h,w["model.norm.weight"],cfg.rms_eps); return int((hn[-1].float()@w["lm_head.weight"].float().t()).argmax())
# static decode buffers
tid=torch.zeros(1,dtype=torch.long,device=dev); ppos=torch.zeros(1,dtype=torch.long,device=dev)
pslot=torch.zeros(1,dtype=torch.int32,device=dev); hout=torch.zeros(1,cfg.hidden_size,dtype=torch.bfloat16,device=dev)
HANDLES=[]; PARAMS=[]
def decode_fwd(capturing):
    h=w["model.embed_tokens.weight"][tid]
    cos,sin=m._rope_cos_sin(ppos); c=cos.unsqueeze(1); s=sin.unsqueeze(1)
    for i in range(cfg.num_layers):
        p=f"model.layers.{i}."; res=h; x=_rms_norm(h,w[p+"input_layernorm.weight"],cfg.rms_eps)
        q=(x@w[p+"self_attn.q_proj.weight"].t()+w[p+"self_attn.q_proj.bias"]).view(1,nh,hd)
        k=(x@w[p+"self_attn.k_proj.weight"].t()+w[p+"self_attn.k_proj.bias"]).view(1,nkv,hd)
        v=(x@w[p+"self_attn.v_proj.weight"].t()+w[p+"self_attn.v_proj.bias"]).view(1,nkv,hd)
        q=q*c+_rotate_half(q)*s; k=k*c+_rotate_half(k)*s; store(i,k,v,pslot)
        knz=kcs[i].view(-1,nkv,hd//NZ,bs,NZ); vnz=vcs[i].view(-1,nkv,hd//NZ,bs,NZ)
        o=torch.empty(1,nh,hd,dtype=h.dtype,device=dev); lse=torch.empty(1,dtype=torch.float32,device=dev)
        if capturing:
            st=torch.npu.current_stream(); torch.npu.graph_task_group_begin(st)
            torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=nh,num_key_value_heads=nkv,input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,actual_seq_qlen=[1],actual_seq_kvlen=[1],out=[o,lse])
            hd_=torch.npu.graph_task_group_end(st); HANDLES.append(hd_); PARAMS.append((q,knz,vnz,o,lse))
        else:
            torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=nh,num_key_value_heads=nkv,input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,actual_seq_qlen=[1],actual_seq_kvlen=[1],out=[o,lse])
        o=o.reshape(1,nh*hd)@w[p+"self_attn.o_proj.weight"].t(); h=res+o
        res=h; x=_rms_norm(h,w[p+"post_attention_layernorm.weight"],cfg.rms_eps)
        g=x@w[p+"mlp.gate_proj.weight"].t(); u=x@w[p+"mlp.up_proj.weight"].t(); h=res+(torch.nn.functional.silu(g)*u)@w[p+"mlp.down_proj.weight"].t()
    hout.copy_(_rms_norm(h,w["model.norm.weight"],cfg.rms_eps))
def upd(ctx):
    st=torch.npu.current_stream()
    for hdl,(q,knz,vnz,o,lse) in zip(HANDLES,PARAMS):
        torch.npu.graph_task_update_begin(st,hdl)
        torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=nh,num_key_value_heads=nkv,input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,actual_seq_qlen=[1],actual_seq_kvlen=[ctx],out=[o,lse])
        torch.npu.graph_task_update_end(st)
ids=tok("The capital of France is").input_ids; P=len(ids)
first=prefill(ids); out=list(ids)+[first]
# warmup decode (eager) to set buffers, then capture
tid.fill_(first); ppos.fill_(P); pslot.fill_(P)
decode_fwd(False)
g=torch.npu.NPUGraph()
with torch.npu.graph(g): decode_fwd(True)
# decode loop via replay
for _ in range(11):
    pp=len(out)-1; tid.fill_(out[-1]); ppos.fill_(pp); pslot.fill_(pp)
    upd(pp+1); g.replay(); torch.npu.synchronize()
    out.append(int((hout[-1].float()@w["lm_head.weight"].float().t()).argmax()))
import time
print("GRAPH-DECODE:", repr(tok.decode(out)))
# A/B: steady-state per-step decode latency, context held fixed (both read same valid cache)
N=300; ctx=P+1
tid.fill_(first); ppos.fill_(P); pslot.fill_(P)
for _ in range(20): upd(ctx); g.replay()
torch.npu.synchronize(); t0=time.perf_counter()
for _ in range(N): upd(ctx); g.replay()
torch.npu.synchronize(); tg=time.perf_counter()-t0
for _ in range(20): decode_fwd(False)
torch.npu.synchronize(); t0=time.perf_counter()
for _ in range(N): decode_fwd(False)
torch.npu.synchronize(); te=time.perf_counter()-t0
print(f"AB B=1 N={N}: graph={tg/N*1000:.3f}ms/step ({N/tg:.0f} step/s)  eager={te/N*1000:.3f}ms/step ({N/te:.0f} step/s)  speedup={te/tg:.2f}x")
