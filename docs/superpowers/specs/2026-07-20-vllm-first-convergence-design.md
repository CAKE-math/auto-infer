# vLLM-Ascend First-Place Convergence Design

## Objective

Make auto-infer rank first against vLLM-Ascend on the shared Qwen3-0.6B
workload for TTFT, B1 TPOT, B16 throughput, load time, equal-capacity peak
allocated memory, and normalized stability without changing generated tokens.

## Acceptance contract

- Run all frameworks sequentially on one idle Ascend 910B1 on `npu2`.
- Use the same model, prompt, BF16 dtype, maximum sequence length, generated
  token count, and warm-up policy.
- Compare memory at the same usable KV token capacity. Report physical scratch
  capacity separately.
- Use at least 20 measured samples for TTFT and throughput stability. Stability
  gates are throughput coefficient of variation and request-elapsed standard
  deviation; absolute throughput standard deviation remains diagnostic only.
- Auto-infer must preserve the existing greedy output digest and pass eager,
  graph, async, preemption, TP, SP, EP, IPC, and service tests.

## Architecture

`GraphPagedRunner` remains the single graph executor. Decode gears retain their
current one-token-per-row shape. A separate `PrefillGear` family is keyed by
the exact total query-token and sequence counts required by FIA-v2 and owns fixed-address token,
position, slot, block-table, cumulative-query-length, KV-length, selected
hidden/logits, and sampled-token buffers. Exact shapes avoid padding work;
executor-owned scratch slots are used only for safe lazy capture.

The prefill graph captures the same `model.forward(ForwardContext)` used by
decode and eager execution. FIA-v2 remains an explicit graph task updated on
the side stream using the existing event pipeline. Unsupported shapes,
chunked batches above the configured token gears, and processor-bearing mixed
batches fail back to the existing eager path. This preserves one model, one KV
cache, and one execution contract.

## Weight and logits ownership

The model loader is authoritative for tied embeddings. When
`tie_word_embeddings` is true, `lm_head.weight` aliases
`model.embed_tokens.weight` even if the checkpoint redundantly stores both.
TP sharding happens before the final alias is installed so both names expose
the correct local tensor.

`BaseCausalLM.logits` no longer creates a persistent FP32 copy of the entire
head. The default fast policy uses the resident BF16 weight and a fixed BF16
graph output. Greedy token parity is tested against the existing FP32 reference
over the acceptance prompt, a deterministic prompt corpus, B1/B16, eager and
graph paths, and at least 256 generated steps. If BF16 changes any winning
token, the implementation uses a bounded mixed-precision candidate refinement
that never materializes a full FP32 head.

## Memory layout

Benchmark configuration specifies usable KV token capacity instead of an
unrelated memory-utilization percentage. Auto-infer reports usable blocks,
scratch blocks, logical weight bytes, and peak allocated bytes. Scratch blocks
are sized to the largest enabled gear, not `max_gear + 1`: lazy capture can
occur while requests already own usable cache blocks, so capture scratch cannot
borrow from the scheduler's live range. The same scratch is reused for padding
after capture. Static logits use the selected logits dtype and prefill stores
only sample-row logits rather than logits for every prompt token.

## Statistics and reporting

`summarize` reports median, mean, standard deviation, coefficient of variation,
and sample count. Throughput reports also derive per-request elapsed samples.
The comparison document ranks stability by CV and elapsed-time deviation while
retaining raw samples and absolute throughput deviation for auditability.

## Failure behavior

- Graph capture or graph-task capability failure disables only the affected
  prefill gear and records the eager fallback counter.
- Precision parity failure prevents the unsafe logits policy from becoming
  default.
- A benchmark with unequal usable KV capacity is rejected rather than ranked.
- Performance acceptance is based on same-run medians; a single favorable run
  cannot pass the gate.

## Implementation order

1. Correct tied-weight ownership and introduce bounded logits policies.
2. Reduce scratch and static-output memory, then prove equal-capacity memory.
3. Add prefill/mixed graph gears and verify first-token parity.
4. Correct benchmark capacity/statistics and run the full three-framework gate.
