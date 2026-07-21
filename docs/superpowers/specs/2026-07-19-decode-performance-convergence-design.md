# Decode Performance Convergence Design

## Objective

Make auto-infer's steady-state Qwen3 decode path at least as fast as Omni-NPU
without weakening correctness, stability, architecture boundaries, or supported
sampling behavior. Implement and measure six changes independently so every
performance claim has a causal before/after result.

## Scope and acceptance workload

The primary performance gate is the existing comparison workload:

- Model: `/data1/models/Qwen3-0.6B`
- Hardware: one Ascend 910B1 on `npu2`
- Data root: `/data2/auto-infer-architecture-20260719`
- Dtype: BF16 weights and activations unless a stage explicitly tests the
  logits-precision boundary
- Maximum model length: 2048
- Prompt: `Explain how a transformer decodes text.`
- Generated tokens: 128, EOS ignored
- Batch gates: B1 TPOT and B16 aggregate token throughput
- Statistics: one warm-up plus at least three measured runs; compare medians and
  retain raw samples

Final steady-state acceptance requires B16 throughput no lower than the
same-run Omni-NPU median and B1 TPOT no higher than the same-run Omni-NPU
median. Startup time and memory are reported separately and must not be hidden
inside the steady-state result.

Correctness acceptance requires:

- greedy sync and async output token streams are identical;
- eager and graph first-token results are identical;
- graph replay is identical across repeated runs;
- the existing host suite passes;
- the existing NPU correctness, concurrency, cancellation, service stability,
  TP, SP, and EP checks pass;
- any lower-precision logits path passes explicit token-parity and numerical
  tolerance tests before it becomes the default.

## Architecture

The scheduler continues to produce immutable `BatchPlan` objects and consumes
`ExecutionResult` objects. Performance state remains executor-owned: graph
gears, input staging buffers, graph-task update streams, device token batches,
and graph outputs do not leak into the scheduler or public API.

The graph runner will have three explicit internal boundaries:

1. `DecodeInputStager` owns persistent pinned-host and device input buffers.
2. `GraphTaskPipeline` owns attention task handles, the update stream, events,
   and per-gear metadata buffers.
3. `DecodeEpilogue` owns logits and sampling policy, selecting a captured greedy
   epilogue or the general sampling implementation.

These units are internal implementation objects. The external `Executor`
protocol remains `submit`, `sampled_of`, `collect_async`, and `collect_result`.

## Stage 1: graph-task update pipeline

Each captured gear receives a dedicated `GraphTaskPipeline`. Attention dynamic
metadata is stored in two slots so the host never mutates metadata still in use
by the preceding replay.

For a decode submission:

1. stage the current batch into the inactive metadata slot;
2. enqueue graph replay on the main stream;
3. enqueue task updates on the gear's update stream;
4. use per-task external events to make each attention task observe its update
   before that task executes;
5. rotate the active metadata slot only after both stream dependencies have
   been recorded.

The first replay is explicitly primed during capture/warm-up. Gear switching
does not share task handles, metadata buffers, or events. Eager fallback remains
single-stream and does not depend on this pipeline.

Failure handling is fail-closed: if the runtime lacks required event or
graph-task-update semantics, graph mode raises a capability error at setup
rather than silently running with stale sequence lengths.

## Stage 2: greedy fast path

`sample_batched` detects an all-greedy batch using host-known sampling metadata.
When no bias, token mask, or penalties are active, it returns `argmax(logits)`
directly and does not execute temperature scaling, softmax, random-number
generation, or multinomial sampling.

Rows with mixed sampling policies continue through the general vectorized
sampler. A request is never moved to the fast path if a processor can change
the winning token.

## Stage 3: persistent input staging

Every gear owns fixed-size pinned-host staging tensors and fixed-address device
tensors for token IDs, positions, slot mappings, sequence lengths, and block
tables. Runtime submission fills host views and uses non-blocking bulk copies.

Block-table staging tracks row lengths and row content. Only changed live rows
and rows transitioning to or from padding are copied. Padding rows always point
to executor-owned scratch blocks. Buffers never reference scheduler-owned
mutable lists after staging finishes.

The eager path reuses a capacity-managed staging object but may resize outside
graph capture. Decode graph device addresses remain stable for the lifetime of
the gear.

## Stage 4: packed projections

Floating-point Qwen-family models store packed projection weights in a model
owned representation:

- QKV: concatenate q, k, and v output rows and execute one matmul, then split
  views into Q, K, and V;
- gate/up: concatenate gate and up output rows and execute one matmul, then feed
  the two views into SwiGLU.

Packing happens once during model loading after TP sharding, so each rank packs
only its local slices. Tied embeddings and lm-head ownership are unaffected.
Dense, eager paged, and graph attention use the same packed projection helpers.

Quantized weights use an explicit quantization-method capability check. A
quantization backend that cannot represent a packed projection retains its
existing correct path instead of being implicitly dequantized or copied.

## Stage 5: captured greedy epilogue

For an all-greedy decode gear without active logits processors, capture the
model forward, logits projection, and argmax into one graph. The gear owns a
fixed-address sampled-token output tensor.

General sampling and processor-bearing requests use the captured model-forward
graph followed by the external `DecodeEpilogue`. Prefill remains eager.

The default logits precision changes only if BF16 or mixed-precision logits pass
both numerical and token-stream parity gates. If they do not, the captured
epilogue retains FP32 accumulation. Performance alone cannot waive this gate.

## Stage 6: async scheduler and token handoff

Replace the per-request device scalar dictionary with immutable
`DeviceTokenBatch` handles. A handle owns one sampled-token tensor plus a
request-ID-to-row mapping. Engine state retains handles, not cloned scalar
tensors.

The next submission groups required rows and performs a vectorized device copy
or gather into the input token buffer. It must not launch one clone and one
scalar assignment per request.

Async D2H uses a pinned CPU output buffer, a dedicated copy stream, and a ready
event. The output worker waits for the event and performs Python materialization
only; it does not issue `.tolist()` against a tensor on the compute stream.

Queue depth remains configurable. Async becomes eligible as a default only when
it is token-identical to sync and does not regress B1 or B16 throughput. If a
workload cannot overlap useful host work, it may select the synchronous path.

## Testing strategy

Every stage follows red-green-refactor:

1. add a host-testable regression for the intended control flow or ownership;
2. run it and confirm the expected failure;
3. implement the smallest production change;
4. run the focused test and full host suite;
5. deploy the isolated tree to `npu2`;
6. run correctness checks before measuring performance;
7. collect paired raw performance samples;
8. keep the stage only if correctness passes and the result is neutral or
   positive within observed run-to-run noise.

NPU-only tests cover stream/event ordering, first-replay priming, alternating
gears, changing sequence lengths, block-table row reuse, greedy graph parity,
and async queue drain/cancellation.

## Rollout and observability

Each optimization has an internal capability flag and a counter identifying
which path ran. Benchmark reports include graph steps, eager steps, captured
greedy steps, general-sampling steps, synchronous collects, and asynchronous
collects.

Stages are integrated in the order listed because later stages depend on the
fixed-address and stream-ownership contracts introduced earlier. A stage that
fails correctness is reverted before continuing. A stage that is correct but
performance-negative remains available only behind an opt-in diagnostic flag
unless a later stage demonstrably changes its cost model.

## Out of scope

- Changing public request or serving APIs
- Adding speculative decoding to the comparison workload
- Quantizing the comparison model
- Copying Omni-NPU or vLLM implementation code
- Trading correctness or supported sampling behavior for benchmark-only speed
- Declaring parity from different output lengths, unreported warm-up, or
  unmatched hardware occupancy
