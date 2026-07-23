# Independent Prefill Gear Design

## Goal

Prevent graph-mode prefill from silently falling back to eager merely because
its flattened token count exceeds the decode batch-size graph limit. Make
Qwen3 profiling compare graph-capable production paths under the same workload.

## Root cause

`ExecutionConfig.max_gear` currently controls two different dimensions:
decode request count and flattened prefill token count. The Qwen3 profile uses
B16 with a nine-token prompt, so prefill contains 144 query tokens. Its
`max_gear=32` setting correctly covers B16 decode but rejects the 144-token
prefill graph and routes that step through `_eager_submit`. vLLM-Ascend keeps
its compiled prefill path enabled, making the current prefill comparison an
eager-versus-compiled comparison.

## Selected architecture

Add `ExecutionConfig.max_prefill_tokens`, defaulting to 256. `max_gear`
continues to mean the maximum captured decode request batch and remains the
MTP request-gear limit. Graph executor construction passes both values to
`GraphPagedRunner`.

The prefill graph family remains one-dimensional and vLLM-compatible:
`[1, 2, 4]`, multiples of eight below 256, then multiples of sixteen. It is
bounded only by `max_prefill_tokens`; sequence count never enters the graph
key. Decode graph enumeration and scratch sizing remain bounded by
`max_gear`. Prefill scratch capacity is bounded independently by
`max_prefill_tokens`.

CLI and serving expose `--max-prefill-tokens` without aliases or compatibility
layers. Graph MTP does not consume the new value because its geometry is
request-count based and its prefill continues through its existing target
runner path.

## Profiling contract

The Qwen3 capture explicitly sets `max_prefill_tokens=256`, which covers its
144-token prefill. Capture metadata records the observed auto-infer path
counters. The capture fails if the profiled prefill did not execute exactly one
prefill graph step or executed an eager prefill step. This prevents a report
from silently publishing another unmatched execution-mode comparison.

The raw trace remains the source of timing and call-stack evidence. After the
change, auto-infer's prefill call stack must contain `prefill-graph` and must
not contain `eager` within the PREFILL phase.

## Correctness and failure behavior

- `max_gear` and `max_prefill_tokens` must both be positive.
- A failed prefill capture remains isolated to that gear and falls back eagerly.
- Runtime never captures a missing gear.
- Oversized prefill still falls back eagerly and increments the existing path
  counter.
- Sampling, KV layout, precision, scheduler behavior, and decode graph behavior
  do not change.

## Acceptance

- Host tests prove independent configuration propagation and that 144 tokens
  select a 144-token prefill gear while decode remains capped at 32 requests.
- The full host suite passes.
- On npu2, auto-infer produces the same Qwen3 output digest as both comparison
  frameworks.
- The auto-infer trace contains one `prefill-graph` call and no `eager` call in
  PREFILL.
- Three raw traces, normalized summary, SVG, Markdown, HTML, and PDF are
  regenerated from the corrected captures.
- The report labels profiled host ranges separately from profiler-free headline
  results and does not infer causality from call-stack depth alone.

## Non-goals

- Changing decode or MTP graph geometry.
- Capturing a two-dimensional prefill graph family.
- Optimizing fused Qwen3 kernels in this change.
- Claiming auto-infer wins prefill before corrected NPU evidence exists.
