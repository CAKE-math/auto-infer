# Zero-Host-Bubble Inter-Step Async Design

## Goal

Make asynchronous decode a real device pipeline rather than an asynchronous
API wrapper. In steady state, when device sampling for step `N` completes,
step `N+1` must already have completed host scheduling, plan construction,
input staging, and graph submission. Its late-bound graph-task update may still
be running, but must finish before the dependent attention node and therefore
cannot open a device bubble.

“Zero host time” means zero host work on the steady-state device critical path;
the CPU still performs work, but that work must be hidden under the preceding
NPU step.

## Required Timeline

```text
Host:
  schedule N+1 ─ plan N+1 ─ stage N+1 ─ submit N+1
                                               └─ dispatch task-update N+1

Compute stream:
  graph N ─ sample N ─ device-token splice N→N+1 ─ graph N+1 early ops
                                                    └─ event wait ─ attention

Copy stream:
                         D2H N ─ client/output finalization

Update stream:
                   graph-task-update N+1 ─ record external event
```

The following ordering conditions are release gates:

```text
schedule_done(N+1) < sample_done(N)
stage_done(N+1)    < sample_done(N)
submit_done(N+1)   < sample_done(N)
task_update_done(N+1) < sample_done(N+1)
device_gap(N,N+1) < 1% of median TPOT
```

## Current Defects

1. `EngineCore._step_async` has a lookahead queue, but the next scheduling
   iteration starts only after `submit`, sampled-token retention, optimistic
   state advance, and async-copy setup for the current step.
2. `GraphPagedRunner` reuses one static `_Gear` buffer set per shape. A second
   in-flight replay can overwrite token IDs, positions, block tables, sequence
   metadata, and captured output while the first replay still consumes them.
   Current correctness therefore depends on submission blocking long enough,
   which defeats async overlap.
3. `GraphTaskPipeline.replay` issues graph replay before all task updates have
   been prepared. Per-layer update calls remain in the current submission’s
   host path.
4. Captured sampled output is retained with a per-step `clone`, adding device
   work and masking the lifetime problem instead of expressing ownership.
5. No trace contract proves that the next step was submitted before the
   current sampling boundary.

## Architecture

### 1. Explicit In-Flight Submission Slots

Introduce a runner-owned slot pool with exactly `async_batches` slots. Each
slot owns independent mutable graph state:

- decode graph gear instances and static input buffers;
- captured sampled-output storage;
- graph-task registration handles;
- metadata update stream and host metadata slots;
- producer/copy/consumer events;
- a monotonically increasing submission sequence.

The same slot is never leased to two in-flight submissions. Synchronous mode
uses one slot and preserves the existing path.

Prefill and mixed batches remain correctness barriers unless they use a
distinct slot-owned captured graph. The initial implementation must prefer a
barrier over unsafe overlap.

### 2. Prepare Then Submit

Split the runner protocol:

```python
prepared = executor.prepare(plan, previous_device_tokens)
handle = executor.submit_prepared(prepared)
```

`prepare` performs host planning and queues fixed-buffer H2D updates on the
leased slot's staging stream. `submit_prepared` places the prepared replay on
the compute stream, then dispatches graph-task updates to a dedicated host
worker and slot-owned update stream.

Replay is deliberately submitted before the dynamic task update. Captured
external events make the graph wait at the affected attention operations,
while early graph work and the preceding step continue on device. The update
must finish before its own graph samples; it need not block replay submission.

### 3. Device-Resident Token Handoff

The token produced by step `N` remains on device and is consumed by the staged
input of step `N+1`. D2H output collection is a second consumer, never a
producer dependency for the next graph.

Captured output lifetime is represented by slot ownership. Per-step
`Tensor.clone()` is forbidden in the async graph path. If a request can skip a
batch, its latest token must be retained in stable device storage whose
lifetime is independent of a graph-output slot.

### 4. Engine Lookahead State

`EngineCore` continues to advance request lengths optimistically, but separates
three states:

- scheduled and prepared;
- submitted and device-in-flight;
- host-finalized.

EOS and stop conditions are finalized asynchronously. Work submitted past a
terminal token may be discarded, but it must not corrupt request/KV state or
be externally emitted.

### 5. Trace Contract

Add host markers for:

- `schedule`;
- `plan`;
- `prepare`;
- `submit`;
- `sample`;
- `finalize`.

The analyzer pairs step `N+1` host markers with step `N` sampling markers and
rejects an async performance artifact unless replay submission precedes the
current sample and the late-bound task update precedes its own sample.
It also reports inter-graph device gaps and refuses to describe async as
zero-host-bubble when the evidence is unavailable.

## Correctness Requirements

- Sync and async greedy tokens are identical for B1, B16, continuous batching,
  request joins/exits, preemption, cancellation, EOS, and stop tokens.
- A slot is not reused until all captured input/output users are complete.
- Static graph buffers from one in-flight submission are never mutated by
  another.
- A skipped request retains its latest device token until it is scheduled.
- D2H and output-thread completion cannot delay the next graph.
- Prefill/mixed batches use a safe barrier until they receive independently
  proven slot ownership.
- MTP remains incompatible with this async mode until it implements the same
  slot and evidence contracts.

## Performance Acceptance

On npu2, using the committed Qwen3 BF16 workload:

- output digest must match synchronous auto-infer;
- async device inter-step gap must be below 1% of median TPOT;
- async TPOT must be lower than sync TPOT;
- async B16 throughput must be higher than sync throughput;
- no sampled-output `clone` may appear in the async graph path;
- host markers must prove `submit_done(N+1) < sample_done(N)`.

If any condition fails, async remains opt-in and the result is reported as an
unfinished performance implementation rather than a completed optimization.
