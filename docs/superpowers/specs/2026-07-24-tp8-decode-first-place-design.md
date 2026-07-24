# TP8 Decode First-Place Design

## Goal

Make Qwen2.5-72B-Instruct BF16 TP8 measurably faster than both vLLM-Ascend and
Omni-NPU without weakening numerical correctness, serving stability, or the
single model/backend seam.

The release gate is:

- identical benchmark topology, prompt corpus, output length, and sampling;
- zero failed or truncated requests;
- graph and paged output parity at the configured numerical tolerance;
- p50 TTFT no worse than vLLM-Ascend;
- at least 5% reproducible throughput lead over the fastest baseline at
  B1, B4, and B16.

The current July 24 TP8 result is the baseline, not a success claim. Auto-infer
already has the best B4/B16 TTFT, but decode ITL is 5--7 times slower than the
fastest baseline. Therefore serving orchestration is not the first optimization
target.

## Root cause

Qwen2.5-72B has 80 decoder layers. Each tensor-parallel layer performs one
all-reduce after attention output projection and one after the MLP down
projection. Eager TP decode therefore exposes 160 collective boundaries per
token, plus the associated Python/CANN launch work. The replicated vocabulary
head also performs the full BF16 projection on every rank.

The existing ACL Graph runner already captures model forward, BF16 lm_head, and
greedy argmax, but TP serving rejects graph mode. Earlier TP graph bring-up
reached `/health` and then stalled on the first request. Logs show that HCCL
capture waited for pending collective work. The architecture currently has no
rank-synchronous capture protocol: decode gears are captured lazily by the first
request, and no TP barrier defines warmup/capture/replay order across ranks.

## Chosen design

### 1. One graph runner, with a TP capture coordinator

Keep `GraphPagedRunner` as the only paged ACL Graph implementation. Add a small
neutral distributed coordination seam:

- enable HCCL graph expansion before any NPU import for TP graph workers;
- expose a no-op-at-TP1 `tp_barrier`;
- prewarm every configured decode gear during executor construction;
- surround warmup and capture of each gear with rank barriers;
- publish worker readiness only after every rank has completed the same gear
  sequence.

The gear sequence is derived from the existing gear policy and `max_gear`; no
model or benchmark batch sizes are hard-coded. Online requests never initiate
graph capture.

`graph_mtp` remains gated for TP. Its target/drafter graph composition requires
a separate numerical gate and is not needed to fix dense Qwen TP decode.

### 2. Preserve replay/update ordering

Each gear continues to own persistent input/output tensors and a
`GraphTaskPipeline`. Replay is submitted on the compute stream; dynamic FIA
metadata is updated on its independent stream against double-buffered metadata.
The TP change does not add host synchronization to the token loop.

The startup coordinator only orders graph construction. Runtime rank ordering
continues to come from the existing deterministic SPMD `BatchPlan` broadcast.

### 3. Optimize only after graph correctness

After graph TP is live and numerically gated, profile the TP8 replay:

1. measure HCCL all-reduce duration and gaps between collectives;
2. verify lm_head and argmax are inside replay;
3. measure graph replay, FIA task update, and D2H token-copy overlap;
4. compare KV capacity at matched HBM settings.

Then apply only evidence-backed changes:

- use capture-safe asynchronous all-reduce where the residual dependency permits
  useful overlap;
- add a vocabulary-parallel greedy head if replicated lm_head is material;
- fuse or coalesce collectives only when the model algebra and BF16 tolerance
  remain unchanged.

Vocabulary parallelism must stay behind `logits_partition`; schedulers and
serving must not learn tensor-parallel vocabulary layout.

## Rejected alternatives

### A separate TP graph runner

Rejected because it would duplicate staging, sampling, KV ownership, graph-task
updates, and async output code. TP is an execution topology, not a second model
forward path.

### Optimize the HTTP serving layer first

Rejected by measurement. Auto-infer already wins B4/B16 TTFT while losing ITL
badly; HTTP and scheduler overhead cannot explain a roughly 160--200 ms decode
step.

### Custom fused collectives before graph liveness

Rejected as an ordering mistake. It increases numerical and deadlock risk while
the larger host-launch bubble remains unremoved.

## Failure handling

- A rank capture exception is reported through the existing replica supervisor;
  the entire replica is terminated.
- Startup readiness is withheld until graph prewarm completes on every rank.
- The watchdog bounds stalled HCCL capture.
- There is no online fallback from a partially constructed TP graph replica to
  eager mode. A failed graph deployment is unhealthy rather than silently
  changing its performance and numerical contract.

## Verification sequence

1. Host unit tests for graph-mode validation, graph worker environment, gear
   enumeration, and rank-synchronous prewarm ordering.
2. Existing full CPU test suite.
3. Qwen3-8B BF16 TP2: startup, first-token liveness, 64-token graph/paged parity,
   continuous batching, and teardown.
4. Qwen2.5-72B BF16 TP8: graph/paged parity and a 30-minute stability soak.
5. Matched three-framework B1/B4/B16 benchmark with at least three measured
   repetitions and p50/CV reporting.
6. TP8 trace-driven ablations until the first-place gate is met.

