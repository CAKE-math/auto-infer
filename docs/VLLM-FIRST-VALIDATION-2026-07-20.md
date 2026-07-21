# vLLM-Ascend First-Place Validation — 2026-07-20

## Scope

The final comparison ran auto-infer, omni-npu 0.14.0, and vllm-ascend 0.20.2
sequentially on an idle Ascend 910B1 device 0 on `npu2`. `npu-smi` was checked
before execution; devices 5 and 6 were occupied by an unrelated TP job and were
not used.

All frameworks used Qwen3-0.6B, BF16, maximum length 2048, the prompt
`Explain how a transformer decodes text.`, greedy decoding with EOS ignored,
128 output tokens, one warm-up, and 20 measured runs. KV capacity was fixed to
14,464 usable tokens. auto-infer used 904 usable 16-token blocks plus 32
executor-owned capture scratch blocks. vLLM-Ascend and Omni-NPU accepted the
same 1,658,847,232-byte KV budget and each runtime logged `GPU KV cache size:
14,464 tokens`; their platform-selected block size was 128.

## Result

| Metric (median unless noted) | auto-infer | omni-npu | vllm-ascend | Winner |
|---|---:|---:|---:|---|
| TTFT proxy | **5.900 ms** | 52.840 ms | 18.992 ms | auto-infer |
| B1 TPOT proxy | **5.530 ms** | 6.519 ms | 17.743 ms | auto-infer |
| B16 throughput | **2,259.2 tok/s** | 1,966.9 tok/s | 847.9 tok/s | auto-infer |
| B1 full request | **0.7083 s** | 0.8807 s | 2.2723 s | auto-infer |
| Engine load and graph setup | **1.484 s** | 52.045 s | 44.482 s | auto-infer |
| Equal-capacity peak torch allocation | **2.7870 GiB** | 9.7314 GiB | 2.7968 GiB | auto-infer |
| Throughput coefficient of variation | **0.700%** | 0.745% | 0.947% | auto-infer |
| B16 request-elapsed standard deviation | **6.48 ms** | 7.77 ms | 23.12 ms | auto-infer |

Absolute throughput standard deviation is retained in raw JSON but is not a
cross-framework stability rank because it scales with mean throughput. The
Both normalized coefficient of variation and absolute request-elapsed
deviation rank auto-infer first.

The post-pipeline async retest was also kept rather than assumed beneficial:
depth 2 preserved the output digest but changed B16 from 2,259.2 to 2,244.1
tok/s, TPOT from 5.530 to 5.941 ms, and throughput CV from 0.700% to 1.850%.
The accepted comparison therefore keeps auto-infer sync scheduling; async
remains correctness-tested and opt-in, not the default performance path.

auto-infer is 68.9% lower in TTFT, 2.66 times faster at B16, and 0.35% lower in
equal-capacity peak torch allocation than vllm-ascend. It is also 14.9% faster
than Omni-NPU at B16 and has 15.2% lower B1 TPOT.

## Correctness and path evidence

- auto-infer and vllm-ascend produced the same 128-token output digest:
  `d23029216ed08f2c`.
- auto-infer preserved the previous 256-token greedy output digest:
  `4ae7f010a6e575e3` after changing the head storage/precision policy.
- Omni-NPU reproducibly produced `f49de48270cac23f`; cross-framework token
  identity is therefore claimed only for auto-infer and vllm-ascend.
- The prefill graph parity script covered B1, B2, a real mixed prefill/decode
  batch with a late request joining an active decode step, an oversized eager
  fallback, and 50 repeated requests. This is the continuous-batching device
  gate, not merely a static multi-prompt batch.
- The 20-run auto-infer report recorded 41 prefill graph steps, 5,094 decode
  graph steps, 5,135 captured greedy steps, and zero external sampler steps for
  the primary workload. B16 prefill correctly used the bounded eager fallback.
- Final host verification passed 231 tests. TP2 and the four-rank
  TP/DP/EP/SP mesh passed; DeepSeek-V2-Lite MLA+MoE graph output matched eager;
  IPC streaming completed ten chunks; and the service stability run completed
  60 requests plus cancellation cleanup.

## Reproduction

auto-infer:

```bash
AUTO_INFER_BENCHMARK_RESULT=/data2/results/auto-infer.json \
  python benchmarks/run_auto_infer.py benchmarks/comparison_manifest.json
```

vllm-ascend:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
AUTO_INFER_BENCHMARK_RESULT=/data2/results/vllm-ascend.json \
python benchmarks/run_vllm.py benchmarks/comparison_manifest.json \
  /data1/models/Qwen3-0.6B vllm-ascend
```

Omni-NPU additionally used `VLLM_PLUGINS=omni-npu,omni_npu_patches` and
`OMNI_NPU_VLLM_PATCHES=ALL` inside `omni_ops_ready:pioneer_liq`.
Its result path was `/data2/results/omni-npu.json`. The final cross-run gate was:

```bash
python benchmarks/compare_results.py \
  /data2/results/auto-infer.json \
  /data2/results/omni-npu.json \
  /data2/results/vllm-ascend.json
```

This aggregation step rejects missing/duplicate frameworks and unequal usable
KV capacity before any ranking is accepted.

Correctness commands included:

```bash
python scripts/verify_prefill_graph.py /data1/models/Qwen3-0.6B
python scripts/verify_greedy_epilogue_graph.py /data1/models/Qwen3-0.6B
python scripts/verify_graph_task_pipeline.py /data1/models/Qwen3-0.6B
python scripts/smoke_graph_engine.py /data1/models/Qwen3-0.6B
```

Raw path counters, phase samples, output digest, capacity, and stability data
are retained under:

- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-auto-infer.log`
- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-omni-npu.log`
- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-vllm-ascend.log`
- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-auto-infer-async.log`

## Conclusion

Under the accepted architecture, correctness, equal-capacity memory, and
stability gates, auto-infer ranks first in every measured latency, throughput,
startup, equal-capacity memory, and stability category.
The decisive TTFT change is exact-shape prefill graph replay; the decisive
memory changes are correct tied-weight ownership, removal of the persistent
FP32 language-model head, resident-dtype graph logits, and bounded scratch.
