import torch, torch_npu
from auto_infer.platform import npu_device
dev=npu_device(0); torch.manual_seed(0)
B,H,nkv,hd,nb,bs = 2,14,2,64,8,128
q=torch.randn(B,H,hd,dtype=torch.bfloat16,device=dev)
bt=torch.tensor([[0],[1]],dtype=torch.int32,device=dev)
clen=torch.tensor([8,8],dtype=torch.int32,device=dev)
scale=hd**-0.5
layouts={
 "3D_nb_bs_nkvhd":(nb,bs,nkv*hd),
 "4D_nb_bs_nkv_hd":(nb,bs,nkv,hd),
 "4D_nb_nkv_bs_hd":(nb,nkv,bs,hd),
}
for name,shp in layouts.items():
    kc=torch.randn(*shp,dtype=torch.bfloat16,device=dev); vc=torch.randn(*shp,dtype=torch.bfloat16,device=dev)
    out=torch.empty(B,H,hd,dtype=torch.bfloat16,device=dev)
    try:
        w=torch_npu._npu_paged_attention_get_workspace(query=q,key_cache=kc,value_cache=vc,num_kv_heads=nkv,num_heads=H,scale_value=scale,block_table=bt,context_lens=clen,out=out)
        torch_npu._npu_paged_attention(query=q,key_cache=kc,value_cache=vc,num_kv_heads=nkv,num_heads=H,scale_value=scale,block_table=bt,context_lens=clen,out=out,workspace=w)
        print(f"{name}: OK out={tuple(out.shape)}")
    except Exception as e:
        print(f"{name}: FAIL {str(e)[:70]}")
