import torch, torch_npu
T, H, qk, vd, nb, bs = 4, 16, 192, 128, 8, 16
dev="npu:0"
q = torch.randn(T, H, qk, dtype=torch.bfloat16, device=dev)
kc = torch.randn(nb, bs, H*qk, dtype=torch.bfloat16, device=dev)
vc = torch.randn(nb, bs, H*vd, dtype=torch.bfloat16, device=dev)
bt = torch.zeros(1, 1, dtype=torch.int32, device=dev)
mask = torch.triu(torch.ones(2048,2048,dtype=torch.int8,device=dev),1)
try:
    out,_ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=kc, value=vc, block_table=bt, input_layout="TND",
        block_size=bs, actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
        num_key_value_heads=H, num_heads=H, scale=qk**-0.5, atten_mask=mask, sparse_mode=3)
    print("FIA_MLA_OK shape:", tuple(out.shape))
except Exception as e:
    print("FIA_MLA_FAILED:", str(e)[:280])
