# Final architecture convergence validation — 2026-07-20

## Outcome

The reviewed inference core now has one production path per supported behavior:
registered execution backends, registered attention families, one shared runner
adapter, one request-lifecycle owner, one two-stage MTP graph pipeline, and one
explicit serving runtime. MTP shape is derived from checkpoint weights through
`MtpGeometry`; there is no production `K, T = 1, 2` or equivalent global shape
constant. Checkpoints with no MTP layers fail capability validation, and models
with unsupported multi-layer recurrence fail explicitly instead of silently
executing the wrong shape.

The final bounded audit found no known duplicate production implementation.
This is backed by zero production `pyflakes` findings, zero duplicate
multi-statement AST bodies, zero internal import cycles, and structural tests
that prevent deleted legacy/fallback symbols and scripts from returning. This
describes the reviewed tree; it does not claim that source minimality can be
proven mathematically or that future dead code is impossible.

## Structural changes

| Area | Final boundary | Removed ambiguity |
|---|---|---|
| Execution | immutable backend specifications + registry + shared `RunnerExecutor` | factory mode switch and duplicated executor forwarding methods |
| Attention | model-declared family resolved by a registry | central GQA/MLA conditional dispatcher |
| Request lifecycle | `EngineCore` owns scheduling, preemption cleanup, completion, and result validation | divergent sync/async/MTP cleanup branches |
| MTP geometry | `MtpGeometry.from_weights()` and `ConfirmedLayout` | global `K/T`, runner-owned layout logic, implicit checkpoint assumptions |
| MTP execution | startup-captured target/drafter graph families only | fused graph, eager fused fallback, online fallback routing |
| Metadata | shared pinned staging, dirty spans/rows, event-ordered side-stream update, two metadata slots | per-step allocation and mutable metadata reuse |
| Serving | injected frozen `ApiRuntime` + `EngineService` | process-global tokenizer/engine/model and empty compatibility subclass |
| P/D | experimental low-level `transfer_hccl` and `copy_blocks`, intentionally unwired from serving | stateful-looking class with unused role state and static-only behavior |
| MTP attention | separately registered GQA recurrent capability; MLA reports unsupported before capture | runner imports of concrete GQA backends |

The production package contains 9,960 physical Python lines across 93 files.
For the checked-out competitor packages, vllm-ascend contains 53,219 lines / 242
files and omni-npu 61,080 / 223. The largest auto-infer runner is 609 lines;
vllm-ascend's main V1 runner is 2,911 lines and omni-npu's NPU runner is 1,377.
These counts are not a quality score by themselves, but they support the lower
indirection and smaller review surface claim.

## Raw runtime data

The Qwen3-0.6B comparison uses one Ascend 910B1, BF16 greedy, batch 16, 128
generated tokens, and equal 14,464-token usable KV capacity.

| Framework | warm TTFT ms | TPOT ms | B16 tok/s | load s | peak GiB | throughput CV |
|---|---:|---:|---:|---:|---:|---:|
| auto-infer | 5.900 | 5.530 | 2,259.2 | 1.484 | 2.7870 | 0.700% |
| omni-npu 0.14.0 | 52.840 | 6.519 | 1,966.9 | 52.045 | 9.7314 | 0.745% |
| vllm-ascend 0.20.2 | 18.992 | 17.743 | 847.9 | 44.482 | 2.7968 | 0.947% |

auto-infer is 14.9% faster than omni-npu and 166.5% faster than vllm-ascend at
B16. Its output digest matches vllm-ascend; omni-npu completed the same length
with a different digest, so only throughput is compared for that pair.

The final Moonlight regression uses the shared Moonlight manifest: BF16 greedy,
32 output tokens, B4 throughput, one warm-up, five measured samples.

| Framework | cold TTFT ms | warm TTFT ms | TPOT ms | B4 tok/s | CV | digest |
|---|---:|---:|---:|---:|---:|---|
| auto-infer final | 31.996 | 25.050 | 13.630 | 228.995 | 0.715% | `599c9e73d403b339` |
| vllm-ascend | 22,717.150 | 47.747 | 42.049 | 91.601 | 0.850% | `599c9e73d403b339` |

auto-infer is 2.50x faster at B4, with 47.5% lower warm TTFT and 67.6% lower
TPOT. Moonlight has `num_nextn_predict_layers=0` and no MTP tensors, so it is an
MLA/MoE graph gate, not an MTP benchmark. Omni-NPU's Moonlight graph run still
has no valid measurement because its rotary graph capture fails on a shape
mismatch.

The final MiMo-7B MTP B4 run produced digest `6e8487eac238fd04`, 32 tokens for
all requests, 79.41% draft acceptance, 1.794 tokens/step, 250.55 tok/s, and
0.509% elapsed-time CV. It performed 114 two-stage graph steps after 5 target
and 10 drafter captures. No fallback counter exists. The matched retained
vllm-ascend result is 176.8 tok/s, so the final architecture remains 41.7%
faster at B4.

The expanded final B16 stability run used ten measured samples: median 895.51
tok/s, 0.240% CV, digest `f660d7348f95bc30`, and 32 tokens for all 16 requests.
It is 32.9% faster than the retained vllm-ascend B16 result of 673.6 tok/s and
has the same digest. An initial five-sample repetition contained one isolated
0.597 s sample and reported 2.38% CV; it is retained separately rather than
discarded. The ten-sample rerun ranged from 0.56899 to 0.57340 s and confirms
that the implementation itself remains below the 1% stability gate.

## Verification evidence

- Final host suite after the 2026-07-21 convergence pass: `416 passed`.
- `python -m compileall -q auto_infer tests benchmarks scripts`: passed.
- Ascend 910B1 BF16 packed-MLA parity on NPU7: maximum absolute error `0.0`
  against independent per-request calls, with identical projected argmax tokens.
- Production `pyflakes`: zero findings.
- Production `vulture --min-confidence 80`: zero findings.
- Duplicate normalized multi-statement function bodies: zero groups.
- Internal `auto_infer` import cycles: zero strongly connected components.
- Moonlight graph counters: seven prefill gears captured, zero capture failures,
  zero online captures, 336 captured greedy steps, zero external sampler steps.
- Continuous-batching, preemption, dirty-row staging, async ownership, graph-task
  event ordering, serving failure recovery, and MTP geometry have dedicated host
  regression tests.
- MLA MTP remains an explicit unsupported capability; P/D remains a low-level
  experimental transfer contract rather than an integrated production topology.

Retained NPU2 evidence:

- `/data2/auto-infer-runtime-convergence-20260721/results/`

- `/data2/auto-infer-ep-dispatch-20260721/logs/final-architecture-20260721/packed-mla-bf16-parity.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/moonlight-auto-final.json`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/moonlight-auto-final.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/mimo-mtp-b4-final.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/mimo-mtp-b16-final.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/mimo-mtp-b16-five-sample-outlier.log`
- `/data2/auto-infer-decode-performance/results/final-20260720/auto-infer.json`
- `/data2/auto-infer-decode-performance/results/final-20260720/omni-npu.json`
- `/data2/auto-infer-decode-performance/results/final-20260720/vllm-ascend.json`

## Competitor conclusion

For the validated inference core, auto-infer leads both competitors on the
matched Qwen latency/throughput/stability gate; it also leads vllm-ascend on the
retained Moonlight MLA/MoE and MiMo MTP measurements. The structural audit found
explicit ownership, registry-based extension seams, immutable execution views,
persistent staging, no internal import cycle, and one graph path without runtime
patch layering or fallback branches. These are scoped findings, not a universal
architecture-quality ordering.

The bounded claim remains important. vllm-ascend and omni-npu support more models,
quantization combinations, connectors, and deployment modes. auto-infer is ahead
for the implemented and measured core; it is not yet ahead on ecosystem breadth.
