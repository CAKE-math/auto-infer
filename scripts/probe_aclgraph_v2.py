import torch, torch_npu
from auto_infer.platform import npu_device
dev=npu_device(0); torch.manual_seed(0)
B,H,nkv,hd,nb,bs,NZ = 2,14,2,64,16,128,16
scale=hd**-0.5
mask=~torch.tril(torch.ones((2048,2048),dtype=torch.bool,device=dev))
kc=torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev)
vc=torch.zeros(nb,bs,nkv,hd,dtype=torch.bfloat16,device=dev)
# populate cache with random KV for L=64 tokens in block 0 and block 1
L=64
def store(blk):
    k=torch.randn(L,nkv,hd,dtype=torch.bfloat16,device=dev); v=torch.randn(L,nkv,hd,dtype=torch.bfloat16,device=dev)
    slot=torch.arange(blk*bs, blk*bs+L, dtype=torch.int32, device=dev)
    kn=kc.view(nb, nkv*hd//NZ, bs, NZ); vn=vc.view(nb, nkv*hd//NZ, bs, NZ)
    torch_npu.npu_scatter_pa_kv_cache(k.contiguous(),v.contiguous(),kn,vn,slot)
store(0); store(1)
q=torch.randn(B,H,hd,dtype=torch.bfloat16,device=dev)
bt=torch.tensor([[0],[1]],dtype=torch.int32,device=dev)
def decode(ctx):  # ctx: list of context lens
    knz=kc.view(-1,nkv,hd//NZ,bs,NZ); vnz=vc.view(-1,nkv,hd//NZ,bs,NZ)
    o=torch.empty(B,H,hd,dtype=q.dtype,device=dev); lse=torch.empty(B,dtype=torch.float32,device=dev)
    torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=H,num_key_value_heads=nkv,
        input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,
        actual_seq_qlen=[1,2],actual_seq_kvlen=ctx,out=[o,lse])
    return o.clone()
e_a=decode([20,20]); e_b=decode([50,50])
# capture at ctx=[20,20]
knz=kc.view(-1,nkv,hd//NZ,bs,NZ); vnz=vc.view(-1,nkv,hd//NZ,bs,NZ)
o=torch.empty(B,H,hd,dtype=q.dtype,device=dev); lse=torch.empty(B,dtype=torch.float32,device=dev)
g=torch.npu.NPUGraph()
with torch.npu.graph(g):
    s=torch.npu.current_stream()
    torch.npu.graph_task_group_begin(s)
    torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=H,num_key_value_heads=nkv,
        input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,
        actual_seq_qlen=[1,2],actual_seq_kvlen=[20,20],out=[o,lse])
    handle=torch.npu.graph_task_group_end(s)
# update to ctx=[50,50] and replay
s2=torch.npu.current_stream()
torch.npu.graph_task_update_begin(s2,handle)
torch_npu.npu_fused_infer_attention_score_v2.out(q,knz,vnz,num_query_heads=H,num_key_value_heads=nkv,
    input_layout="TND",softmax_scale=scale,block_table=bt,block_size=bs,sparse_mode=3,atten_mask=mask,
    actual_seq_qlen=[1,2],actual_seq_kvlen=[50,50],out=[o,lse])
torch.npu.graph_task_update_end(s2)
g.replay(); torch.npu.synchronize()
d_b=(o.float()-e_b.float()).abs().max().item(); d_a=(o.float()-e_a.float()).abs().max().item()
print(f"ACLgraph-v2 replay vs eager[50]={d_b:.4f} vs eager[20]={d_a:.4f} ->", "DYNLEN_OK" if d_b<d_a and d_b<0.5 else "FAIL")
