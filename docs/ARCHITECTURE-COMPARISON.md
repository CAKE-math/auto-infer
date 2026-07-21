# auto-infer vs omni-npu vs vllm-ascend

Comparison date: 2026-07-20. The runtime experiment used one free Ascend 910B1
on npu2 and the shared [comparison manifest](../benchmarks/comparison_manifest.json):
Qwen3-0.6B, BF16, max length 2048, greedy/ignore-EOS, 128 output tokens, batch 16,
one warmup and 20 measured runs. Each framework ran sequentially on device 0
with exactly 14,464 usable KV tokens.

The auto-infer and vllm-ascend outputs had the same 128-token digest. omni-npu
completed 128 tokens but had a different digest, so its throughput is comparable
while cross-framework token identity is not claimed.

## Runtime results

| Metric (median unless noted) | auto-infer | omni-npu 0.14.0 | vllm-ascend 0.20.2 |
|---|---:|---:|---:|
| TTFT proxy, one-token offline request | **5.900 ms** | 52.840 ms | 18.992 ms |
| TPOT proxy, B1 128-token request | **5.530 ms** | 6.519 ms | 17.743 ms |
| B16 throughput | **2,259.2 tok/s** | 1,966.9 tok/s | 847.9 tok/s |
| B1 full request | **0.7083 s** | 0.8807 s | 2.2723 s |
| Engine load + graph preparation | **1.484 s** | 52.045 s | 44.482 s |
| Equal-capacity peak torch allocation | **2.7870 GiB** | 9.7314 GiB | 2.7968 GiB |
| Throughput coefficient of variation | **0.700%** | 0.745% | 0.947% |
| Request-elapsed standard deviation | **6.48 ms** | 7.77 ms | 23.12 ms |

After full convergence, auto-infer ranks first in every latency, throughput,
startup, equal-capacity memory, and stability row. It is 14.9% faster than
omni-npu and 2.66× faster than vllm-ascend at B16. Its TTFT is 68.9% lower than
vllm-ascend. Peak memory
is now directly comparable because every runtime logged the same 14,464-token
usable KV capacity.

The runners persist clean JSON through `AUTO_INFER_BENCHMARK_RESULT`; the final
`benchmarks/compare_results.py` gate rejects missing frameworks and unequal
runtime-reported usable-token capacity.

A final auto-infer depth-2 async rerun was token-identical but slower and more
variable than sync for this workload, so the reported winner uses the stable
sync default. Async remains available for workloads where measurement proves a
benefit.

Retained final raw logs on npu2:

- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-auto-infer.log`
- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-omni-npu.log`
- `/data2/auto-infer-decode-performance/logs/final-20260720/compare-vllm-ascend.log`

The omni image required its documented patch plugin configuration
(`VLLM_PLUGINS=omni-npu,omni_npu_patches` and
`OMNI_NPU_VLLM_PATCHES=ALL`). Without it, both Qwen2 and Qwen3 failed during
attention-backend lookup. With it, Qwen3 completed the benchmark. This setup
dependency is part of the architecture evaluation rather than discarded noise.

## Architecture comparison

Production package size is 9,960 physical Python lines / 93 files for
auto-infer, 61,080 / 223 for omni-npu, and 53,219 / 242 for vllm-ascend.
Size alone is not quality, but it quantifies the amount of code
one must traverse when combined with the dependency structure below.

| Dimension | auto-infer after convergence | omni-npu | vllm-ascend |
|---|---|---|---|
| Core control flow | Direct `EngineCore → BatchPlan → Executor → ExecutionResult`; registered backends share one runner adapter; graph pipeline is composed from staging, replay/update, and epilogue units | vLLM flow plus environment-selected runtime patch application | vLLM engine plus platform/worker/model-runner specialization |
| Mutable ownership | Engine/service thread owns request, scheduler, and KV state; immutable batch-token owners replace scalar clones | Ownership crosses upstream classes and applied patches | Mostly inherited from vLLM, with Ascend worker/runner state |
| Model/backend coupling | Model declares GQA/MLA family; central registry selects dense/paged/graph | Per-model implementations plus patches and extra config | Broad vLLM model reuse, with Ascend attention and runner branches |
| Serving/IPC lifecycle | One service, broker, explicit close/join, request-id demux, and request-local injected API runtime with no process-global engine/tokenizer | Mature vLLM serving, modified by patches | Mature vLLM serving |
| Parallel configuration | One named TP/DP/EP/CP/SP mesh, host and HCCL tested | Rich topology/parallel tuning, high configuration surface | Mature vLLM parallel configuration and broad deployment support |
| Failure mode observed here | Fail-fast contract errors; all phase-one runs completed | Missing patch env caused model-load failure; `ALL` also logged a duplicate scheduler-patch conflict | Benchmark completed; long compile/startup path |
| Decode/prefill critical path | Exact-shape prefill graph plus decode replay; event-released side-stream metadata updates; double-buffered metadata; persistent dirty-row staging; packed projections; resident-dtype captured head/argmax; one two-stage MTP path | Graph runtime and async scheduler supplied through Omni patches | Ascend compilation/ACL graph integrated into the vLLM runner |
| Extension breadth | Two attention families and a small model registry; recurrent MTP is GQA-only, and P/D remains an unwired low-level interface | Broad optimized model/operator catalog | Broadest upstream model/API ecosystem |

## Verdict

For the scoped inference core, auto-infer now has lower indirection, clearer
ownership, a smaller model-to-runtime extension seam, and a faster accepted
decode path than both plugin stacks. The performance work did not add backend
branches to the engine: graph-task pipelining, input staging, epilogue capture,
and token transfer remain separately testable components. This meets the stated
quality bar for an independent framework rather than another vLLM patch layer.

It is not categorically superior as a complete product: both baselines still
have substantially broader model, quantization, deployment, and ecosystem
coverage. The evidence-backed conclusion is:

- architecture audit: auto-infer has no internal import cycle and keeps the
  reviewed engine, worker, attention, distributed, and serving boundaries
  acyclic; the competitor rows describe their checked-out control-flow shape,
  not a universal quality ranking;
- startup, TTFT, B1 TPOT, B16 throughput, equal-capacity memory, and both stability measures: auto-infer leads in this test;
- optimized-model and deployment breadth: omni-npu leads;
- mature ecosystem breadth: vllm/vllm-ascend leads.

The new framework passes the structural gates defined for this scoped core and
the matched runtime gate. An unconditional whole-product “better than both”
claim would still exceed the evidence.

Detailed raw samples, path counters, commands, and known limitations are in
[the first-place validation report](VLLM-FIRST-VALIDATION-2026-07-20.md). The
post-review structural gates and final NPU regressions are in
[the final architecture validation](FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md).
