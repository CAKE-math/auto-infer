import torch, torch_npu
dev="npu:0"
T,H,hd,nb,bs = 8,14,64,64,16
q=torch.randn(T,H,hd,dtype=torch.bfloat16,device=dev)
kc=torch.randn(nb,bs,2*hd,dtype=torch.bfloat16,device=dev)
vc=torch.randn(nb,bs,2*hd,dtype=torch.bfloat16,device=dev)
bt=torch.arange(T,dtype=torch.int32,device=dev).view(T,1)
mask=torch.triu(torch.ones(2048,2048,dtype=torch.int8,device=dev),1)
# try actual_seq_lengths as device tensors
cu_q=torch.arange(1,T+1,dtype=torch.int32,device=dev)
kvl=torch.full((T,),5,dtype=torch.int32,device=dev)
for kind,aq,ak in [("tensor",cu_q,kvl),("list",list(range(1,T+1)),[5]*T)]:
    try:
        o,_=torch_npu.npu_fused_infer_attention_score(query=q,key=kc,value=vc,block_table=bt,
            input_layout="TND",block_size=bs,actual_seq_lengths=aq,actual_seq_lengths_kv=ak,
            num_key_value_heads=2,num_heads=H,scale=hd**-0.5,atten_mask=mask,sparse_mode=3)
        print(f"FIA actual_seq_lengths as {kind}: OK shape={tuple(o.shape)}")
    except Exception as e:
        print(f"FIA as {kind}: FAIL {str(e)[:100]}")
