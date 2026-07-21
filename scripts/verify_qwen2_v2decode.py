"""Step 1: Qwen2 prefill+decode via nano-vllm recipe (NZ cache + scatter_pa + FIA-v2),
eager (no graph yet). Verify greedy == existing plain forward (Paris...)."""
import torch, torch_npu
from transformers import AutoTokenizer
from auto_infer.models.qwen2 import Qwen2Model, _rms_norm, _rotate_half
from auto_infer.platform import npu_device
dev=npu_device(0); path="/data0/models/Qwen2.5-0.5B-Instruct"
tok=AutoTokenizer.from_pretrained(path); m=Qwen2Model.from_pretrained(path,dev,torch.bfloat16)
cfg=m.cfg; w=m.w; NZ=16
nh,nkv,hd=cfg.num_heads,cfg.num_kv_heads,cfg.head_dim; scale=hd**-0.5
nb,bs=64,128
mask=~torch.tril(torch.ones((2048,2048),dtype=torch.bool,device=dev))
kcs=[torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev) for _ in range(cfg.num_layers)]
vcs=[torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev) for _ in range(cfg.num_layers)]
def qkv(x,i):
    p=f"model.layers.{i}.self_attn."
    q=(x@w[p+"q_proj.weight"].t()+w[p+"q_proj.bias"]).view(-1,nh,hd)
    k=(x@w[p+"k_proj.weight"].t()+w[p+"k_proj.bias"]).view(-1,nkv,hd)
    v=(x@w[p+"v_proj.weight"].t()+w[p+"v_proj.bias"]).view(-1,nkv,hd)
    return q,k,v
def store(i,k,v,slot):
    kn=kcs[i].view(nb,nkv*hd//NZ,bs,NZ); vn=vcs[i].view(nb,nkv*hd//NZ,bs,NZ)
    torch_npu.npu_scatter_pa_kv_cache(k.contiguous(),v.contiguous(),kn,vn,slot)
def layer(i,h,pos,slot,bt,prefill,ctx):
    p=f"model.layers.{i}."; res=h
    x=_rms_norm(h,w[p+"input_layernorm.weight"],cfg.rms_eps)
    q,k,v=qkv(x,i); T=q.shape[0]
    cos,sin=m._rope_cos_sin(pos); c=cos.unsqueeze(1); s=sin.unsqueeze(1)
    q=q*c+_rotate_half(q)*s; k=k*c+_rotate_half(k)*s
    store(i,k,v,slot)
    if prefill:
        o=torch_npu.npu_fused_infer_attention_score_v2(q,k,v,num_query_heads=nh,num_key_value_heads=nkv,
            input_layout="TND",softmax_scale=scale,sparse_mode=3,atten_mask=mask,
            actual_seq_qlen=[T],actual_seq_kvlen=[T],next_tokens=0)[0].view(T,nh,hd)
    else:
        knz=kcs[i].view(-1,nkv,hd//NZ,bs,NZ); vnz=vcs[i].view(-1,nkv,hd//NZ,bs,NZ)
        o=torch.empty(T,nh,hd,dtype=h.dtype,device=dev); lse=torch.empty(T,dtype=torch.float32,device=dev)
        torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=nh,num_key_value_heads=nkv,
            input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,
            actual_seq_qlen=[1],actual_seq_kvlen=ctx,out=[o,lse])
    o=o.reshape(T,nh*hd)@w[p+"self_attn.o_proj.weight"].t(); h=res+o
    res=h; x=_rms_norm(h,w[p+"post_attention_layernorm.weight"],cfg.rms_eps)
    g=x@w[p+"mlp.gate_proj.weight"].t(); u=x@w[p+"mlp.up_proj.weight"].t()
    return res+(torch.nn.functional.silu(g)*u)@w[p+"mlp.down_proj.weight"].t()
def run(ids):
    P=len(ids); bt=torch.tensor([[0]],dtype=torch.int32,device=dev); out=list(ids)
    # prefill
    h=w["model.embed_tokens.weight"][torch.tensor(ids,device=dev)]
    pos=torch.arange(P,device=dev); slot=torch.arange(P,dtype=torch.int32,device=dev)
    for i in range(cfg.num_layers): h=layer(i,h,pos,slot,bt,True,None)
    h=_rms_norm(h,w["model.norm.weight"],cfg.rms_eps); nxt=int((h[-1].float()@w["lm_head.weight"].float().t()).argmax()); out.append(nxt)
    # decode
    for _ in range(11):
        pp=len(out)-1; h=w["model.embed_tokens.weight"][torch.tensor([out[-1]],device=dev)]
        pos=torch.tensor([pp],device=dev); slot=torch.tensor([pp],dtype=torch.int32,device=dev)
        for i in range(cfg.num_layers): h=layer(i,h,pos,slot,bt,False,[pp+1])
        h=_rms_norm(h,w["model.norm.weight"],cfg.rms_eps); out.append(int((h[-1].float()@w["lm_head.weight"].float().t()).argmax()))
    return out
ids=tok("The capital of France is").input_ids
print("v2decode:", repr(tok.decode(run(ids))))
