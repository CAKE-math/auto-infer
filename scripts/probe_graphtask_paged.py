import torch, torch_npu
from auto_infer.platform import npu_device
dev=npu_device(0); torch.manual_seed(0)
B,H,nkv,hd,nb,bs = 2,14,2,64,4,128
q=torch.randn(B,H,hd,dtype=torch.bfloat16,device=dev)
kc=torch.randn(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev)
vc=torch.randn(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev)
bt=torch.tensor([[0],[1]],dtype=torch.int32,device=dev)
clen=torch.tensor([8,8],dtype=torch.int32,device=dev)
out=torch.empty(B,H,hd,dtype=torch.bfloat16,device=dev)
scale=hd**-0.5
def ws():
    return torch_npu._npu_paged_attention_get_workspace(query=q,key_cache=kc,value_cache=vc,
        num_kv_heads=nkv,num_heads=H,scale_value=scale,block_table=bt,context_lens=clen,out=out)
def eager(lens):
    clen.copy_(torch.tensor(lens,device=dev)); o=torch.empty_like(out)
    w=torch_npu._npu_paged_attention_get_workspace(query=q,key_cache=kc,value_cache=vc,
        num_kv_heads=nkv,num_heads=H,scale_value=scale,block_table=bt,context_lens=clen,out=o)
    torch_npu._npu_paged_attention(query=q,key_cache=kc,value_cache=vc,num_kv_heads=nkv,num_heads=H,
        scale_value=scale,block_table=bt,context_lens=clen,out=o,workspace=w)
    return o.clone()
e8=eager([8,8]); e40=eager([40,40])
clen.copy_(torch.tensor([8,8],device=dev))
w=ws()
g=torch.npu.NPUGraph(); s=torch.npu.current_stream()
with torch.npu.graph(g):
    torch.npu.graph_task_group_begin(s)
    torch_npu._npu_paged_attention(query=q,key_cache=kc,value_cache=vc,num_kv_heads=nkv,num_heads=H,
        scale_value=scale,block_table=bt,context_lens=clen,out=out,workspace=w)
    handle=torch.npu.graph_task_group_end(s)
us=torch.npu.Stream()
clen.copy_(torch.tensor([40,40],device=dev)); w2=ws()
with torch.npu.stream(us):
    torch.npu.graph_task_update_begin(us,handle)
    torch_npu._npu_paged_attention(query=q,key_cache=kc,value_cache=vc,num_kv_heads=nkv,num_heads=H,
        scale_value=scale,block_table=bt,context_lens=clen,out=out,workspace=w2)
    torch.npu.graph_task_update_end(us)
g.replay(); torch.npu.synchronize()
d40=(out.float()-e40.float()).abs().max().item(); d8=(out.float()-e8.float()).abs().max().item()
print(f"graphtask replay-vs-eager40={d40:.4f} vs-eager8={d8:.4f} ->", "DYNLEN_OK" if d40<d8 and d40<0.5 else "FAIL")
