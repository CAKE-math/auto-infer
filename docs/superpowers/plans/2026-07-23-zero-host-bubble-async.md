# Zero-Host-Bubble Async Decode Implementation Plan

> Execute this plan in the existing `decode-performance` worktree. The target
> is greedy BF16 graph decode; MTP remains explicitly incompatible and
> prefill/mixed execution is a drain barrier until it owns equivalent slots.

## Acceptance contract

In steady-state decode, host scheduling, immutable plan construction, device
input staging, graph-task update, and graph submission for step `N+1` must
finish before device sampling for step `N` finishes. Sync and async tokens must
match, graph/static buffers must never be concurrently reused, EOS/abort must
not free leased KV, and the async path must contain no sampled-output clone.

## Task 1: Lock the engine and executor contracts with failing tests

**Files**

- Modify: `tests/test_async_output_thread.py`
- Modify: `tests/test_engine_core.py`
- Modify: `tests/test_graph_task_pipeline.py`
- Add: `tests/test_async_execution_slots.py`
- Modify: `auto_infer/engine/executor.py`

**Steps**

1. Add a recording executor whose first output future stays unresolved.
2. Assert that prepare and submit for batch `N+1` occur before collecting `N`.
3. Add EOS and abort tests proving KV remains allocated until every submitted
   batch lease is finalized.
4. Add a history-dependent sampling rejection test for async mode: optimistic
   placeholders cannot safely feed repetition/frequency/presence processors.
5. Add graph-pipeline tests asserting `update -> replay`, immutable metadata
   ownership, and configurable slot depth.
6. Add slot-pool and stable-token-store CPU tests: unique active slots, reuse
   only after release, skipped-request token retention, and no clone.
7. Run the focused tests and confirm they fail for the intended reasons.

## Task 2: Make submitted work lifetime-safe in EngineCore

**Files**

- Modify: `auto_infer/engine/engine_core.py`
- Modify: `auto_infer/engine/scheduler.py`
- Modify: `auto_infer/engine/executor.py`
- Modify: `tests/test_async_output_thread.py`
- Modify: `tests/test_engine_core.py`

**Steps**

1. Replace tuple queue entries with an `InFlightBatch` record holding scheduler
   output, device handle, output future, request leases, and per-batch metrics.
2. Count a request lease for every submitted batch.
3. Split scheduler disposal into `retire_request` (remove from admission sets)
   and `reclaim_request` (register valid prefix, free KV, delete request).
4. On EOS, stop, max-token completion, or abort, retire immediately but reclaim
   only after the last batch lease drains.
5. Normalize `num_computed_tokens` after EOS lookahead truncation before prefix
   registration.
6. Move async TTFT/token accounting to actual oldest-batch finalization.
7. Drain one completed batch, then refill the queue before returning so serving
   work cannot create the next inter-step host bubble.
8. Run engine-focused tests.

## Task 3: Add explicit graph execution slots and stable device tokens

**Files**

- Add: `auto_infer/worker/async_slots.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/decode_input_stager.py`
- Modify: `auto_infer/worker/prefill_input_stager.py`
- Modify: `auto_infer/worker/async_output.py`
- Modify: `auto_infer/executor_backends.py`
- Modify: `auto_infer/engine/executor.py`
- Modify: `tests/test_async_execution_slots.py`
- Modify: `tests/test_architecture_convergence.py`

**Steps**

1. Pass `async_batches` and `max_num_seqs` into graph runner construction.
2. Give each in-flight decode slot independent captured graph gears, static
   device inputs/outputs, pinned host staging, registrations, update stream,
   and a monotonically increasing submission ID.
3. Add a runner-owned fixed-capacity `DeviceTokenStore` with stable request
   rows. Scatter sampled tokens D2D on the compute stream and return refs into
   the store; a skipped request keeps its row unchanged.
4. Add persistent per-gear device/pinned index buffers for the D2D scatter.
5. Delete async sampled-output `clone()` and source-protection backpressure.
   Keep each output slot leased until its D2H future is consumed.
6. Add executor request-release notification so token rows are reclaimed only
   with the corresponding engine request.
7. Make `RunnerExecutor.supports_async()` delegate to the runner. Only the
   graph runner advertises the new contract; paged/eager execution does not.
8. Treat prefill/mixed and first-time capture as explicit drain barriers.
9. Run slot/store/protocol tests.

## Task 4: Split graph-task preparation from replay

**Files**

- Modify: `auto_infer/worker/graph_task_pipeline.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `tests/test_graph_task_pipeline.py`
- Modify: `tests/test_graph_decode_runner.py`

**Steps**

1. Replace `replay(graph, ctx)` with `prepare(ctx) -> ticket` and
   `submit(graph, ticket)`.
2. Bind registrations and metadata storage to the owning execution slot,
   removing host mutation of shared `backend.reg` during submission.
3. Queue graph-task updates on the slot update stream before replay is
   submitted; preserve ordering through the captured external event.
4. Set metadata ring depth from `async_batches`, not a hard-coded two.
5. Split graph runner `prepare(plan, previous_tokens)` from
   `submit_prepared(prepared)`. Preparation leases and stages a slot;
   submission only queues the ready graph and token-store scatter.
6. Return the slot on every exception path.
7. Run graph pipeline/runner tests.

## Task 5: Add evidence gates

**Files**

- Add: `auto_infer/engine/async_timeline.py`
- Add: `scripts/analyze_async_timeline.py`
- Add: `tests/test_async_timeline.py`
- Modify: `benchmarks/profile_qwen3.py`
- Modify: `benchmarks/run_auto_infer.py`

**Steps**

1. Emit opt-in low-overhead markers for schedule, plan, prepare, submit,
   sample-ready, D2H-ready, and finalize, carrying submission IDs.
2. Parse Chrome/torch-profiler JSON and pair `submit(N+1)` with
   `sample(N)`.
3. Report host ordering violations and p50/p95 device inter-graph gaps.
4. Reject the “zero-host-bubble” label if markers are absent, if any steady
   ordering condition fails, or if `aten::clone` occurs in async decode.
5. Run analyzer unit tests.

## Task 6: Local convergence verification

**Steps**

1. Run focused tests for engine, executor, graph staging, graph task pipeline,
   output copies, timeline analysis, and continuous batching.
2. Run the entire CPU test suite.
3. Run static searches for `reused_output`, async clone, shared registration
   swaps, and hard-coded metadata depth.
4. Review the diff for unused compatibility layers or duplicate paths and
   remove them.

## Task 7: npu2 `/data2` validation and iteration

**Steps**

1. Recheck free NPU/container state and deploy the exact committed source into
   a fresh directory under `/data2`.
2. Run Qwen3 BF16 sync and async B1/B16, 128-token decode, 20 measured samples,
   plus staggered continuous batching, EOS, cancellation, and request skip.
3. Compare output digests and fail immediately on any correctness mismatch.
4. Capture Chrome traces containing prefill and multi-step decode.
5. Verify every steady-state `submit_end(N+1) < sample_end(N)`, no async clone,
   and p50/p95 device gaps below 1% median TPOT.
6. Require async TPOT lower than sync and B16 throughput higher than sync.
7. If a gate fails, use trace evidence to fix and rerun; keep async disabled by
   default until every gate passes.

## Task 8: Documentation and delivery

**Files**

- Modify: `docs/PRODUCTION_STABILITY_AND_ACCURACY.md`
- Modify: performance report sources if and only if npu2 measurements pass

**Steps**

1. Document the exact supported boundary: graph decode, greedy BF16,
   prefill/mixed barriers, MTP incompatible.
2. Record raw commands, machine topology, source commit, trace artifact paths,
   correctness digests, median/CV, TPOT/throughput, and ordering verdict.
3. Run final verification once more.
4. Commit the implementation and evidence as one coherent change.
