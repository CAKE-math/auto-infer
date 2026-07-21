# MiMo MTP and Moonlight validation — 2026-07-20

## Scope and checkpoint capability

The requested Moonlight checkpoint is
`/data2/models/Moonlight-16B-A3B-Instruct`. Its model metadata is
`DeepseekV3ForCausalLM`, `num_nextn_predict_layers=0`, and its weight index
contains zero MTP/next-token-prediction tensors. MTP cannot truthfully be
enabled for this checkpoint in any runtime.

Moonlight was therefore used as the requested MLA/MoE persistent-engine
correctness gate. The full paged `EngineCore` completed coherently on npu2:

```text
paged-engine gen: ' Paris, which is located in the northern part of the country'
=== V3 (Moonlight) through full paged EngineCore ===
COHERENT
```

## Moonlight performance

Moonlight performance is reported separately from the MTP comparison because
this checkpoint has no trained MTP layer. The matched workload is one Ascend
910B1, BF16 greedy decoding, a 512-token model limit, 32 generated tokens,
batch 4 for throughput, one warm-up, and five measured runs. Both successful
runtimes produced the same 32-token output digest (`599c9e73d403b339`).

| Runtime | Cold post-init TTFT | Warm B1 TTFT | B1 TPOT | B4 tok/s | B4 CV | Peak allocated | Load time | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| auto-infer graph | **32.02 ms** | **25.57 ms** | **13.46 ms** | **233.78** | **0.53%** | 33.16 GiB | **19.65 s** | Valid; seven prewarmed prefill gears |
| vLLM-Ascend | 22,717.15 ms | 47.75 ms | 42.05 ms | 91.60 | 0.85% | **30.60 GiB** | 78.98 s | Valid; runtime-managed async |
| Omni-NPU | — | — | — | — | — | — | — | Failed before measurement |

For this workload auto-infer reaches 2.55x vLLM-Ascend's median B4 throughput,
reduces TPOT by 68.0%, and lowers warmed TTFT by 22.18 ms. Peak allocation
remains 2.56 GiB higher. Omni-NPU fails during graph warm-up/capture in rotary
embedding with a shape mismatch (`16` versus `8192`), so it has no valid
performance result.

Cold TTFT is now measured identically by both committed runners: one B1,
one-token request immediately after construction and before any explicit
warm-up. Both runners submit the same pre-tokenized input IDs, while tokenizer
initialization and the first prompt encoding are charged to load time. The raw
samples are 32.02 ms for auto-infer and 22.717 s for
vLLM-Ascend. vLLM-Ascend defers substantial ACL graph compilation/first replay
work into that first request; the earlier table omitted this phase and therefore
could not support a cold-to-cold comparison. Both result JSON files carry the
scalar and the matching single-element raw sample.

MLA prefill now uses the same one-dimensional flattened-token capture policy as
vLLM and Omni-NPU. With `max_gear=32`, startup captures exactly
`[1, 2, 4, 8, 16, 24, 32]`; all seven captures succeeded and measured requests
performed zero online captures. The first post-initialization request therefore
replays a graph instead of paying the previous 424 ms lazy-capture cost. The
final benchmark reported 12 prefill graph replays, 6 intentional fallbacks, 0
capture failures, 0 online captures, and separate dirty-copy totals of 7
prefill rows and 85 decode rows. A staggered mixed prefill/decode probe completed
token-identically under graph and forced-eager execution. Its counters reported
54 prefill graph replays and zero capture failures. The B4 benchmark's 40-token initial prefill is
intentionally above `max_gear=32` and uses eager fallback; subsequent decode
steps remain graphed.

Retained final evidence on npu2:

- `results/moonlight-auto-prefill-final.json`
- `results/moonlight-vllm-cold-final.json`
- `logs/final-prefill-20260720/moonlight-auto-prefill-final.log`
- `logs/final-prefill-20260720/moonlight-vllm-cold-final.log`
- `logs/final-prefill-20260720/moonlight-continuous-graph-eager.log`
- `logs/final-prefill-20260720/qwen-prefill-graph.log`

The manifest requests 4096 usable KV tokens and the measured B4/512 workload
does not approach that limit. vLLM nevertheless reports an 8576-token physical
KV cache for the requested 256 MiB while the comparison adapter normalizes the
reported usable capacity to 4096; this capacity-accounting difference should
be fixed before treating a memory-capacity comparison as strictly matched.
The exact workload is saved in `benchmarks/moonlight_manifest.json`.

Real MTP comparison uses `/data1/models/MiMo-7B-Base`, whose checkpoint has
one trained next-token-prediction layer and 16 MTP tensors.

## Correctness and stability

- The final two-stage path produced 32 tokens for every request. Graph capture
  is startup-only and failure is fatal; the retired fused fallback no longer
  exists in production.
- B4 is token-identical to auto-infer graph-plain (`6e8487eac238fd04`).
  B16 is token-identical to same-run vLLM-Ascend MTP
  (`f660d7348f95bc30`) across all 16 requests.
- MTP acceptance was 79.4%, or 1.794 emitted tokens per verify step.
- Five-sample CV was 0.03% at B4 and 0.27% at B16.
- Transformers, auto-infer, vLLM-Ascend, and Omni-NPU can diverge at a
  near-tied greedy token under BF16/FIA. The first three B4 request streams were
  identical between auto-infer and vLLM-Ascend; the fourth diverged later.
  This is not an MTP rejection error: auto-infer MTP remains identical to its
  non-MTP graph baseline.
- The known near-tie remains batch-shape dependent: B16 graph-plain uses digest
  `dabdb0f57f92c586`, while both auto-infer MTP and vLLM-Ascend MTP use
  `f660d7348f95bc30`. B4 auto-infer MTP matches graph-plain, while vLLM differs
  on the same fourth-prompt near-tie. Transformers BF16 produces a third digest.
  This is a numerical-path difference, not speculative acceptance corruption;
  every accepted token is produced by the target graph and rejection remains
  branchlessly enforced.

## Matched comparison

Single Ascend 910B1, BF16, greedy, ignore EOS, max model length 512, four base
prompts, 32 output tokens, one exact-batch warm-up followed by five measured
runs. Throughput uses the median elapsed time.

| Runtime | B4 tok/s | B4 CV | Load time | Status |
|---|---:|---:|---:|---|
| auto-infer graph-MTP | **254.5** | **0.03%** | 6.6 s | First; zero fallback |
| vLLM-Ascend MTP | 176.8 | 0.69% | 43.3 s | Runs |
| Omni-NPU MTP eager | 101.1 | 1.33% | 55.0 s | Runs only with eager fallback |

Omni-NPU graph MTP cannot initialize this checkpoint: graph capture invokes
`aclnnNonzeroV2`, which attempts to synchronize a captured stream and exits
with error 107027.

The throughput-oriented repeated-prompt B16 probe was:

| Runtime | B16 tok/s | Median elapsed | CV |
|---|---:|---:|---:|
| auto-infer graph-MTP | **913.2** | **0.5607 s** | **0.27%** |
| auto-infer graph-plain | 813.7 | 0.6293 s | one parity sample |
| vLLM-Ascend MTP | 673.6 | 0.7601 s | 0.96% |

The final raw auto-infer elapsed samples were:

- B4: `0.50314, 0.50276, 0.50295, 0.50287, 0.50298` seconds.
- B16: `0.56051, 0.56066, 0.56038, 0.56075, 0.56437` seconds.

The post-architecture-convergence B16 stability rerun used ten samples and
reported 895.5 tok/s with 0.240% CV and the same `f660d7348f95bc30` digest.

The final architecture statistics were identical at both batch sizes after one
warm-up plus five measured generations: 5 target captures, 10 drafter captures,
and only two-stage graph steps. Final single-token
steps use qd=1 to preserve KV progress at block boundaries. They still execute
the MTP hidden layer so completed full blocks remain safe for prefix reuse, but
skip the unused final norm, language head, and draft transfer. Correct
per-request long-prefill threshold semantics reduced each measured B16
generation from 25 eager scheduling rounds and 40 graph rounds to 2 eager rounds
and 20 graph rounds. Batched prefill/first-decode then executes one target
forward and one MTP forward per scheduler round and projects only final rows.

## Conclusion

The production MiMo MTP path is first in the matched B4 and B16 comparison.
It leads same-run vLLM-Ascend by 44.0% at B4 and 35.6% at B16, exceeding the
1.03x acceptance gate with CV below 1%. The decisive defect was continuous
batching: `long_prefill_token_threshold` had incorrectly capped the aggregate
prefill batch instead of each request, serializing B16 admission. The final
implementation also separates target verification from compacted drafting,
uses startup-only graph families, dirty metadata staging, event-ordered pinned
copies, one packed result transfer, and batched prefill/MTP heads.

Final retained logs on npu2:

- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/auto-b4-kv-safe-final.log`
- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/auto-b16-kv-safe-final.log`
- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/vllm-b4-final.log`
- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/vllm-b16-final.log`
- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/plain-graph-b16-parity.log`
- `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720/hf-greedy-final.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/mimo-mtp-b4-final.log`
- `/data2/auto-infer-decode-performance/logs/final-architecture-20260720/mimo-mtp-b16-final.log`
