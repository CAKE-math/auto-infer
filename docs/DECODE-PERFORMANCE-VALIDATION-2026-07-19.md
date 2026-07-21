# Decode performance convergence validation — 2026-07-19

## Acceptance result

The scoped goal passed on one Ascend 910B1 on `npu2`, using the shared
`benchmarks/comparison_manifest.json` (Qwen3-0.6B, BF16, max length 2048,
greedy, 128 generated tokens, B1/B16, one warm-up, three measured samples).
Devices 0–4 and 7 were free before the run; devices 5–6 were occupied and were
not used. The occupancy snapshot is retained as
`/data2/auto-infer-decode-performance/logs/final-20260719/npu-smi-before-bench.log`.

| Median metric | auto-infer | omni-npu 0.14.0 | vllm-ascend 0.20.2rc1 |
|---|---:|---:|---:|
| TTFT proxy | 41.796 ms | 52.500 ms | **19.129 ms** |
| B1 TPOT | **5.804 ms** | 6.354 ms | 17.981 ms |
| B1 full request | **0.779 s** | 0.860 s | 2.303 s |
| B16 throughput | **2,185.5 tok/s** | 1,911.6 tok/s | 857.8 tok/s |
| Engine load / preparation | **1.764 s** | 52.059 s | 61.715 s |
| Peak torch allocation | 9.198 GiB | 8.583 GiB | **2.797 GiB** |
| B16 throughput stdev | 29.74 | 24.65 | **13.08** |

Relative to Omni-NPU, auto-infer is 14.3% faster at B16 and has 8.7% lower B1
TPOT. It therefore passes both performance gates. Relative to vLLM-Ascend it is
2.55× faster at B16 and has 67.7% lower B1 TPOT.

The auto-infer and vLLM-Ascend output digest is `d23029216ed08f2c` for 128
tokens. Omni-NPU produced 128 tokens with output digest `f49de48270cac23f`.
The Omni throughput comparison is valid for the matched greedy length, but
cross-framework token identity with Omni is not claimed.

## Raw measurements

| Framework | TTFT samples (s) | B1 full samples (s) | B16 samples (tok/s) |
|---|---|---|---|
| auto-infer | 0.039560, 0.041796, 0.042420 | 0.783621, 0.769275, 0.778920 | 2139.620, 2195.377, 2185.454 |
| omni-npu | 0.049228, 0.052981, 0.052500 | 0.845141, 0.859513, 0.863213 | 1879.677, 1911.621, 1928.159 |
| vllm-ascend | 0.019129, 0.019460, 0.018915 | 2.372641, 2.302685, 2.241894 | 838.039, 862.777, 857.797 |

The final auto-infer path counters were: 776 graph steps, 11 eager/prefill
steps, 776 captured-greedy steps, zero external-sampler steps, captured gears
`[1, 16]`, 476 dirty block-table rows and 60,928 copied block-table elements.
The benchmark recorded sync scheduling (`async_mode.enabled=false`, configured
queue depth 2).

## Optimization attribution

| Stage | B16 median tok/s | B1 TPOT |
|---|---:|---:|
| Pre-fix baseline | 1,197.9 | 10.106 ms |
| Replay/update pipeline | 1,640.3 | 7.655 ms |
| Greedy argmax fast path | 1,805.0 | 7.053 ms |
| Persistent staging / dirty rows | 2,035.0 | 5.739 ms |
| Packed QKV and gate/up | 2,051.3 | 6.406 ms |
| Captured FP32 head and argmax | 2,236.5 | 5.822 ms |
| Final fresh three-framework run | 2,185.5 | 5.804 ms |

Stage-to-stage measurements are diagnostic rather than additive; thermal state
and run order introduce normal variance. The final comparison is the acceptance
number.

Async scheduling was remeasured after the device pipeline fixes. In the first
three-sample sweep, depth 3 was +15.6% at B1 and +0.45% at B16. A five-sample
confirmation found -4.6% at B1 and +4.37% at B16. Because no single depth was
non-regressing at both gates, sync remains the default and async stays opt-in.

## Correctness and stability matrix

Fresh validation on the final source included:

| Gate | Result |
|---|---|
| Qwen3 dense vs HF first token | match; packed weights asserted |
| Paged FIA vs plain dense, three prompts | all match |
| Graph task pipeline, alternating gears and long replay | pass; 490 graph steps |
| Captured greedy epilogue, 256-token repeats and sampler fallback | pass; 641 graph steps |
| Paged and graph async vs sync | all four streams match |
| Graph async 48 concurrent requests | pass |
| Cancellation followed by 60-request reuse | pass |
| Forced KV preemption/recompute | pass; one real eviction |
| Shutdown with pending event-backed copies | pass, no deadlock |
| Persistent IPC | 10 streamed tokens, clean close |
| Service cancellation/reuse | 60 completions after cancellation, clean close |
| Config-driven SP2×EP2 four-card mesh | pass |
| DeepSeek graph vs eager FIA-v2 | all four streams match |

Qwen3 TP2 now runs after fixing biasless TP sharding, but its longer BF16
sequence diverges from TP1 after an identical initial segment. This is recorded
as an open strict-token-parity limitation, not a pass. DeepSeek graph teardown
also prints a CANN `model not in current ctx` warning when destroying the
second-device graph; generation parity passes and the process exits zero.

The earlier phase-one SP2×EP2 model-token, Moonlight paged, and routing checks
remain retained under `/data2/auto-infer-architecture-20260719/logs/`; they were
not affected by the Qwen packed-projection changes.

## Commands and retained logs

Local verification:

```bash
pytest -q
python -m compileall -q auto_infer benchmarks scripts tests
git diff --check
```

Representative final NPU commands (all run inside `auto-infer-dev-20260624`,
with the indicated `ASCEND_RT_VISIBLE_DEVICES`):

```bash
python scripts/smoke_qwen3.py /data1/models/Qwen3-0.6B
python scripts/smoke_paged.py /data1/models/Qwen3-0.6B
python scripts/verify_graph_task_pipeline.py /data1/models/Qwen3-0.6B
python scripts/verify_greedy_epilogue_graph.py /data1/models/Qwen3-0.6B
python scripts/verify_async_sched.py /data1/models/Qwen3-0.6B
torchrun --standalone --nproc_per_node=2 scripts/smoke_tp_qwen2.py /data1/models/Qwen3-0.6B
torchrun --standalone --nproc_per_node=4 scripts/verify_parallel_mesh_npu.py
python scripts/verify_deepseek_graphdecode.py /data1/models/DeepSeek-V2-Lite-Chat
python scripts/verify_ipc_serving.py
python scripts/verify_service_stability.py
python benchmarks/run_auto_infer.py benchmarks/comparison_manifest.json
```

Omni-NPU ran in `omni_ops_ready:pioneer_liq` with
`VLLM_PLUGINS=omni-npu,omni_npu_patches`,
`OMNI_NPU_VLLM_PATCHES=ALL`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Both external frameworks used:

```bash
python benchmarks/run_vllm.py benchmarks/comparison_manifest.json \
  /data1/models/Qwen3-0.6B <framework-name>
```

Full framework logs and all fresh correctness logs are under
`/data2/auto-infer-decode-performance/logs/final-20260719/`.

Environment versions: torch-npu 2.10.0, vLLM 0.20.2+empty,
vllm-ascend 0.20.2rc1, Omni-NPU/vLLM 0.14.0, CANN/driver as reported by
`npu-smi 25.5.0`.
