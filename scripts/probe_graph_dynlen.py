import torch, torch_npu
from auto_infer.platform import npu_device
dev=npu_device(0); torch.manual_seed(0)
H,hd,bs = 14,64,128
q=torch.randn(1,H,hd,dtype=torch.bfloat16,device=dev)
kc=torch.randn(1,bs,2*hd,dtype=torch.bfloat16,device=dev)   # H kv heads? use 2 kv heads
vc=torch.randn(1,bs,2*hd,dtype=torch.bfloat16,device=dev)
bt=torch.zeros(1,1,dtype=torch.int32,device=dev)
mask=torch.triu(torch.ones(2048,2048,dtype=torch.int8,device=dev),1)
aq=torch.tensor([1],dtype=torch.int32,device=dev)
akv=torch.tensor([8],dtype=torch.int32,device=dev)   # kv-len, will update
def fia():
    o,_=torch_npu.npu_fused_infer_attention_score(query=q,key=kc,value=vc,block_table=bt,
        input_layout="TND",block_size=bs,actual_seq_lengths=aq,actual_seq_lengths_kv=akv,
        num_key_value_heads=2,num_heads=H,scale=hd**-0.5,atten_mask=mask,sparse_mode=3)
    return o
akv.copy_(torch.tensor([8],device=dev)); eager8=fia().clone()
akv.copy_(torch.tensor([40],device=dev)); eager40=fia().clone()
# capture at kv-len=8
akv.copy_(torch.tensor([8],device=dev))
g=torch.npu.NPUGraph()
with torch.npu.graph(g): out=fia()
# replay after updating kv-len tensor to 40
akv.copy_(torch.tensor([40],device=dev)); g.replay()
d_reads = (out.float()-eager40.float()).abs().max().item()   # small if re-reads tensor
d_baked = (out.float()-eager8.float()).abs().max().item()    # small if baked at capture
print(f"replay-vs-eager40(re-read)={d_reads:.4f}  replay-vs-eager8(baked)={d_baked:.4f}")
print("RE-READS TENSOR (graph dyn-len OK)" if d_reads<d_baked else "BAKED (needs graph_task_update)")
