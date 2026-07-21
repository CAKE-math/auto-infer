# MLA Prefill Graph TTFT Design

## Goal

Make Moonlight's first-request and warmed B1 TTFT lower than the measured
vLLM-Ascend baseline of 46.58 ms without weakening output correctness,
continuous batching, chunked prefill, or graph-capture failure isolation.

## Root cause

`GraphMlaBackend` implements graph capture and dynamic graph-task updates but
does not advertise `supports_prefill_graph`. `GraphPagedRunner` consequently
routes every Moonlight prefill through `_eager_submit`; only decode uses graph
replay. The original benchmark confirms this with zero prefill graph steps and
17 eager steps.

A same-device runtime experiment enabled the existing path without modifying
the source. Median B1 TTFT fell from 69.89 ms to 28.20 ms and the sampled token
remained identical. Synchronous capture of the first exact shape took 423.88
ms, so merely enabling the capability fixes warmed TTFT but makes cold TTFT
worse.

## Selected architecture

### Capability

Both graph GQA and graph MLA backends explicitly support prefill graph replay.
The runner continues to query the backend capability instead of branching on
model names or architectures.

### Covering gears

Prefill graphs use the same one-dimensional flattened-token sizing policy as
vLLM and Omni-NPU. The default sizes are `[1, 2, 4]`, then multiples of 8 below
256, then multiples of 16. Sizes are truncated at `max_gear`. With the current
`max_gear=32`, the complete family is `[1, 2, 4, 8, 16, 24, 32]`: seven graphs,
not a Cartesian product with sequence count. Raising `max_gear` to 256 produces
the same 35-size family observed in the vLLM-Ascend Moonlight run.

At runner initialization, every enabled token gear is captured before the
engine accepts requests. The startup work is reported as model load time rather
than being charged to the first request. There is no synchronous online capture
for a missing key. Work above `max_gear` uses the existing eager fallback.

The static graph buffers have `query_gear` token rows and capacity for
`query_gear` sequence metadata rows. Runtime graph-task updates carry the real
TND cumulative query lengths and KV lengths and pass the persistent block-table
buffer as the exact `[:B]` row view required by FIA, so sequence count does not
enter the graph key. An NPU probe replayed one captured 16-token graph with both
one and two sequence metadata rows successfully. The implementation must retain
this device acceptance test and must not restore a two-dimensional graph family.

### Padding and KV isolation

The fixed graph has `query_gear` rows. If the real batch has fewer query tokens,
the stager represents the padding rows as one independent dummy sequence. Its
query/KV lengths, block table, and slot mappings are scratch-only, so real
sequence metadata and live KV capacity never change. Unused block-table and
sampling rows are zeroed or scratch-backed in the persistent maximum-capacity
buffers.

Each final real token's row is recorded before the dummy sequence and remains
the row used for vocabulary projection and sampling. Dummy outputs are never
returned. The live sequences' future decode slots and block tables are therefore
unchanged and are populated normally by later decode steps.

The graph-task update receives exactly `B` real block-table rows when there is
no padding and `B + 1` rows when the scratch dummy is present. The selected
vLLM-compatible gear adds at most 15 padding rows, so one 16-token scratch block
covers the dummy under the default policy. Scratch capacity remains bounded by
`max_gear` as part of runner initialization.

### Input staging

`PrefillInputStager.stage` accepts a real query count and sequence count less
than or equal to its token gear. It preserves persistent pinned host and NPU
buffers, updates only dirty block-table rows, returns both real and padded query
counts, and builds the first `B` sampling entries exclusively from real rows.
The captured lm-head and argmax operate over fixed-size selected rows; only the
first `B` results are exposed to the engine.

### Failure behavior

Startup capture is isolated per token gear. A failed gear is recorded in
`failed_prefill_gears` and serves eagerly; failure of one shape does not abort
model initialization. Runtime never retries a known failure and never performs
an unbounded or surprise capture. The bounded cache limit equals the number of
configured token gears.

## Data flow

1. Model and paged KV caches load.
2. The runner enumerates the vLLM-compatible flattened-token gears and attempts
   capture.
3. A request batch selects the smallest covering token gear; sequence count is
   runtime metadata, not part of the key.
4. The persistent stager copies real inputs, appends scratch-only padding, and
   updates dirty block rows.
5. Graph-task metadata is updated on the independent stream and the graph
   replays.
6. Captured BF16 lm-head plus argmax returns only real requests' sampled tokens.
7. Oversized or failed gears use the existing eager path.

## Correctness requirements

- Graph and eager Moonlight outputs must be token-identical for B1 and B4 over
  32 generated tokens.
- Padding must not change live KV contents or subsequent decode output.
- Mixed prefill/decode continuous batches must select correct sample rows and
  preserve request ordering.
- Chunked prefill emits no token before the real prompt completes.
- A failed prefill gear must fall back to eager without poisoning other gears.

## Performance and stability acceptance

- On one Ascend 910B1 with the committed Moonlight manifest, both the first
  post-initialization request and warmed B1 median TTFT must be below 46.58 ms.
- Five warmed runs must report TTFT CV below 5%.
- B4 throughput must not regress by more than 3% from 225.93 tok/s.
- Startup time and peak allocated memory are reported explicitly; startup must
  remain below the measured vLLM-Ascend load time of 69.07 s.
- The result must show nonzero `prefill_graph_steps`, no online capture during
  measured requests, and the same output digest as the eager reference.
- With `max_gear=32`, exactly seven prefill token gears are attempted; graph
  count must not multiply by sequence count.

## Files and boundaries

- `auto_infer/layers/attention/mla.py`: declare the MLA backend capability.
- `auto_infer/worker/graph_decode_runner.py`: select covering gears, prewarm the
  bounded graph family, isolate failed captures, and expose path counters.
- `auto_infer/worker/prefill_input_stager.py`: stage real rows plus scratch-only
  padding without changing the runner or scheduler interfaces.
- `tests/test_graph_decode_runner.py`: vLLM-compatible token-gear selection,
  prewarm, and graph policy.
- `tests/test_prefill_input_stager.py`: padding, sample-row, dirty-copy, and KV
  isolation behavior.
- `tests/test_attention_registry.py`: backend capability regression.
- `docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md`: final NPU measurements and
  correctness evidence.

## Non-goals

- Prefill graphs above the configured `max_gear`.
- Changing the async scheduler.
- Changing MLA math, MoE routing, numerical precision, or decode graphs.
- Hiding graph warmup outside reported engine load time.
