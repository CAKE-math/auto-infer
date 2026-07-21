# MiMo MTP First-Place Design

Date: 2026-07-20
Branch: `perf/decode-performance`
Target: one Ascend 910B1 on `npu2`

## Goal

Make auto-infer's MiMo-7B MTP path faster than vLLM-Ascend under the same
single-device BF16 greedy workload at both batch 4 and batch 16, without
changing the emitted greedy token stream. The accepted result must lead the
same-run vLLM-Ascend median throughput by at least 3% at both batch sizes.

## Evidence and Root Cause

The retained baseline is:

| Runtime | B4 tok/s | B16 tok/s |
|---|---:|---:|
| auto-infer graph-MTP | 164.2 | 222.1 |
| vLLM-Ascend MTP | 172.8 | 657.1 |
| auto-infer graph-plain | — | 364.1 |

MiMo supplies one trained MTP layer and the matched vLLM configuration uses one
speculative token. Auto-infer currently captures target verification, target
sampling, rejection, a two-position MTP layer, and a second two-position
language head in one graph. Each request therefore executes the MTP path for
both target positions even when the second position is rejected. It also ties
the target and drafter to the same request-count gear and prevents either phase
from using its natural flattened-token shape.

Retained experiments falsify the smaller host-side hypotheses. Combining the
three result copies, flattening the output buffer, and projecting only a
gathered drafter head did not improve B4. Synchronous graph-task updates also
regressed plain B16. Those changes alter the tail of the step but leave the
unconditional two-position drafter execution intact.

vLLM-Ascend instead treats target verification and proposal as distinct device
phases. Its proposer prepares confirmed token/hidden-state rows after target
sampling and runs the drafter on those rows. The auto-infer design will adopt
that phase boundary while retaining its smaller scheduler and graph runner.

## Architecture

### 1. Target verify graph

One target graph family is keyed by flattened target query tokens. For K=1,
an active request contributes two rows: the last confirmed token and the
carried draft. The graph performs:

1. target model forward over `2 * request_count` rows;
2. model-resident BF16 language head and greedy argmax for both rows;
3. device-side comparison of the carried draft with the first prediction;
4. construction of `accepted_count`, confirmed-row indices, confirmed token
   IDs, and per-request output lengths.

The graph retains target pre-final-norm hidden states in a persistent buffer.
No target prediction is copied to the host between target and drafter phases.

### 2. Confirmed-row compaction

A graph-captured compaction epilogue converts the fixed two-row target layout
into packed confirmed rows. A rejected request contributes its first target
row. An accepted request contributes both target rows. The packed buffers are:

- confirmed target hidden states;
- confirmed token IDs used by the MTP input projection;
- confirmed absolute positions and slot mappings;
- cumulative query lengths and KV lengths for the MTP attention update;
- the final packed row for each request, used to select its next draft.

The packed capacity is `2 * request_gear`; its active length is
`request_count + accepted_count`. Padding rows use only the runner's reserved
scratch blocks. Compaction and metadata production stay on device except for
the Python sequence-length lists required by the current FIA graph-task update
API. A persistent pinned control buffer copies only the `request_count`
acceptance flags after the target event; the host derives the exact cumulative
query and KV-length lists from those flags. It does not copy target predictions
or hidden states. Those lists are staged through the existing double-buffered
update pipeline and never trigger graph capture.

### 3. Drafter graph

The MTP graph family is keyed by flattened confirmed-row count, not request
count. It performs the trained MiMo MTP layer over the compacted rows, updates
only confirmed MTP KV positions, projects only the last confirmed row of each
request through the shared model-resident BF16 language head, and writes one
next draft per request.

Target and drafter use independent graph-task registrations, persistent input
buffers, and gear selection. The target completion event orders compaction and
drafter replay on device. Runtime execution is lookup-only: every accepted gear
is captured at runner construction, and unsupported captures are recorded and
fall back to the current correct fused graph for that gear.

### 4. Packed result handoff

After the drafter graph completes, one persistent packed result tensor contains
the two target predictions, accepted count, and next draft for each active
request. One nonblocking copy moves the active prefix to a pinned host slot.
The host materializes request dictionaries only after the copy event completes.
This result transfer is separate from the small inter-phase acceptance-control
copy. No vocabulary-sized tensor and no intermediate hidden state crosses D2H.

### 5. Engine contract

`EngineCore` keeps its existing speculative contract: it receives one or two
confirmed tokens per request and one next draft. The runner owns all target,
compaction, drafter, KV, graph, and copy details. Scheduler semantics,
preemption invalidation, prefix caching, chunked prefill, and exact output-length
truncation remain unchanged.

## Components and Boundaries

- `graph_mtp_runner.py` coordinates the target and drafter phases and exposes
  the existing `execute_spec_mtp(BatchPlan) -> ExecutionResult` interface.
- A dedicated MTP staging component owns persistent target, compacted-drafter,
  metadata, and packed-output buffers. It has no scheduler mutation authority.
- `GraphTaskPipeline` remains the only abstraction allowed to update dynamic FIA
  tasks. Target and drafter receive independent pipeline instances.
- The rejection sampler semantics remain greedy prefix acceptance. Device
  compaction changes layout only, never acceptance decisions.
- Benchmark scripts own comparison methodology and result validation; runtime
  code contains no benchmark-only branch.

## Correctness Invariants

1. Target KV may contain both verification positions during a step, matching
   ordinary speculative verification.
2. MTP KV receives only confirmed positions. A rejected draft never becomes
   visible to later MTP attention.
3. Per-request confirmed rows remain in chronological order after compaction.
4. The next draft is projected from the final confirmed row for that request.
5. Padding reads and writes only reserved scratch blocks.
6. Requests finishing after their first emitted token ignore the second token
   and next draft without corrupting another request's row.
7. Runtime graph selection and replay never capture a new graph.
8. MTP output must be token-identical to auto-infer graph-plain for every
   acceptance pattern exercised by the tests and NPU gate.

## Failure Handling

- A target or drafter gear that fails startup capture is marked unavailable.
  Requests for that gear use the existing fused MTP graph, which remains the
  correctness fallback.
- A batch larger than the largest target gear is split by request count before
  target verification. Each chunk completes both target and drafter phases
  before its result is merged.
- Invalid compacted counts, row indices, or scratch mappings raise before replay
  in host tests and fail the affected execution in device validation. They are
  not silently clamped.
- Async scheduling is not enabled for MTP as part of this change. The device
  phase split must first pass synchronous correctness and performance gates.

## Testing

### Host tests

- rejected, accepted, and mixed acceptance compaction layouts;
- chronological packed positions and per-request final-row indices;
- MTP KV excludes rejected rows;
- target and drafter choose independent covering gears;
- startup prewarm, capture-failure fallback, and zero runtime capture;
- one packed D2H result layout;
- B4/B16 padding and scratch isolation;
- oversized chunking, continuous batching, preemption, chunked prefill, and
  exact output caps;
- full existing test suite.

### npu2 correctness gates

- MiMo MTP versus auto-infer graph-plain for the four base prompts at B4;
- repeated-prompt B16 parity;
- forced all-accept, all-reject, and mixed-accept diagnostic batches;
- staggered continuous batching with late arrivals;
- capture attempts occur only at startup and capture failures remain zero for
  the accepted production gears.

### Performance gates

Run auto-infer and vLLM-Ascend sequentially on the same idle Ascend 910B1 with
the same MiMo checkpoint, prompt list, BF16 greedy settings, 32 output tokens,
one exact-shape warmup, and five measured runs. For both B4 and B16:

- auto-infer median throughput is at least 1.03 times the same-run
  vLLM-Ascend median;
- auto-infer throughput coefficient of variation is at most 3%;
- every output has exactly 32 tokens;
- auto-infer MTP digest equals its same-run graph-plain digest;
- acceptance rate and tokens per step are reported;
- per-phase target, compaction, drafter, D2H, and total step timings are retained.

The final report must include raw samples and retained npu2 log paths. A faster
result that violates parity, uses a different speculative depth, or omits either
B4 or B16 does not pass.

## Non-Goals

- Increasing speculative depth beyond K=1.
- Enabling the asynchronous scheduler for MTP.
- Changing MiMo weights, precision, prompts, or acceptance semantics.
- Optimizing Moonlight, which has no trained MTP layer.
- Claiming superiority from load time alone.
