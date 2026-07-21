# auto-infer NPU dev environment (npu2)

## Container
- name: `auto-infer-dev-20260624`  (8x 910B1, torch_npu device_count=8)
- image: `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.20.2rc1`
- created with: 8x `/dev/davinci0-7` + davinci_manager/devmm_svm/hisi_hdc + driver mounts + `-v /data0,1,2`
- repo path (host): `/data2/npubench/10138/auto-infer`  (host-editable, container-visible)

## Versions (inside container)
- Python 3.11.15, torch 2.10.0, torch_npu 2.10.0, CANN latest
- vllm 0.20.2+empty, vllm_ascend 0.20.2rc1 (source ref: `/vllm-workspace/vllm-ascend/vllm_ascend`)

## Bring-up model
- `/data0/models/Qwen2.5-0.5B-Instruct` (Qwen2 dense GQA: hidden 896, 24 layers, heads 14, kv_heads 2, rope_theta 1e6, tie_word_embeddings, bf16, vocab 151936)

## NPU paged attention reference (from vllm-ascend attention_v1.py)
- KV cache shape per layer: `(2, num_blocks, block_size, num_kv_heads, head_size)` (idx0=K, idx1=V)
- main attn op: `torch_npu.npu_fused_infer_attention_score(query, key, value, block_table, input_layout="TND", block_size, actual_seq_lengths, actual_seq_lengths_kv, num_key_value_heads, num_heads, scale, atten_mask=, sparse_mode=)`
- write KV: `torch_npu._npu_reshape_and_cache(...)` by slot_mapping
- RoPE: `torch_npu.npu_rotary_mul`
- decode-only paged: `torch_npu._npu_paged_attention`

## Plan 2 milestone 1 (decided)
Real `NpuExecutor`: single-card eager Qwen2.5-0.5B forward WITH real paged attention
(host KVCacheManager block tables -> NPU paged KV cache -> FIA), verify coherent output + parity vs HF transformers.
