# TP8 Decode First-Place Implementation Plan

**Goal:** Enable capture-safe TP ACL Graph execution, validate numerical and
serving correctness on NPU, then remove the measured TP8 decode bottlenecks
until Qwen2.5-72B beats both comparison frameworks.

**Architecture:** Reuse `GraphPagedRunner` and the existing SPMD serving
control plane. Add only a distributed capture-coordination seam; keep TP,
graph, sampling, and serving concerns in their current owners.

**Release rule:** No performance result is accepted unless parity, liveness,
request accounting, and stability gates pass first.

---

## Task 1: Turn TP graph policy into a tested capability

**Files:**

- Modify: `tests/test_tp_server.py`
- Modify: `auto_infer/serving/tp_server.py`

1. Replace the old test that rejects all graph modes with tests that accept
   `graph`, continue to reject `graph_mtp`, and require the HCCL graph expansion
   environment only for graph workers.
2. Run the focused test and confirm it fails for the old policy.
3. Implement the minimum configuration/environment change.
4. Run the focused test and full TP server tests.

## Task 2: Add rank-synchronous graph prewarm

**Files:**

- Modify: `tests/test_graph_decode_runner.py`
- Modify: `tests/test_parallel_state.py`
- Modify: `auto_infer/distributed/parallel_state.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`

1. Add host tests for the decode gear sequence and a coordinator that brackets
   each gear capture with barriers.
2. Add a no-op-at-TP1 `tp_barrier`.
3. Prewarm decode gears at construction when graph execution is enabled.
4. Ensure online `_get_gear` is lookup-only in TP mode; absence is a startup
   failure, not a lazy collective capture.
5. Keep single-rank lazy capture behavior unchanged where useful.
6. Run focused graph/distributed tests.

## Task 3: Verify locally before NPU deployment

**Files:** no production changes

1. Run formatting/static sanity checks (`git diff --check`, compile).
2. Run the complete CPU test suite.
3. Inspect the diff for duplicated runner logic, model-specific branches, and
   serving-to-worker leakage.

## Task 4: TP2 graph liveness and numerical gate on npu2

**Deployment:** `/data2/auto-infer-tp-graph-20260724/source`

1. Stop only the currently identified TP8 auto-infer rank processes.
2. Deploy the exact committed source state.
3. Launch Qwen3-8B BF16 TP2 graph on two 910B devices.
4. Verify graph prewarm finishes before `/health`.
5. Send a real 64-token request and continuous-batching burst.
6. Run the same inputs in paged TP2 and compare tokens/logits under the existing
   BF16 tolerance.
7. Exercise clean shutdown and one-worker failure supervision.

## Task 5: Qwen2.5-72B TP8 graph gate

1. Launch BF16 graph TP8 with the production topology.
2. Verify all eight ranks load the intended weight shards and complete the same
   graph gear sequence.
3. Run single-request parity against paged TP8.
4. Run prefix reuse and mixed arrival traffic.
5. Run a 30-minute stability soak; require zero errors, timeouts, truncations,
   and rank divergence.

## Task 6: Profile and optimize the remaining decode critical path

**Likely files, selected only after trace evidence:**

- `auto_infer/distributed/parallel_state.py`
- `auto_infer/models/base.py`
- `auto_infer/models/qwen2.py`
- `auto_infer/layers/attention/gqa.py`
- `auto_infer/worker/graph_decode_runner.py`

1. Capture TP8 prefill and multi-step decode traces.
2. Attribute each token to graph replay, HCCL, lm_head, graph-task update,
   staging, sampling, and D2H.
3. Add a failing test for one measured optimization at a time.
4. Prefer, in order:
   - capture-safe collective overlap;
   - vocabulary-parallel greedy head behind `logits_partition`;
   - collective coalescing/fusion where algebra permits.
5. Re-run parity and stability after every optimization.

## Task 7: Three-framework acceptance benchmark

1. Match model, BF16, TP8, devices, prompt corpus, output length, sampling,
   warmup, KV capacity, and memory utilization.
2. Run at least three measured repetitions for B1/B4/B16.
3. Report throughput, TTFT, ITL, CV, failures, HBM, and exact command/source
   revisions.
4. Accept only if auto-infer is at least 5% faster than the best baseline in all
   three throughput tiers while preserving the correctness gates.
5. Store raw Chrome-readable traces and JSON results beside the report.

