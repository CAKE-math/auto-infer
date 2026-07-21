# Decode Performance Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove auto-infer's decode critical-path overhead and meet or exceed Omni-NPU on the matched Qwen3 B1/B16 workload without correctness or stability regressions.

**Architecture:** Keep `BatchPlan`/`ExecutionResult` and public serving APIs stable. Add executor-owned graph-task pipelining, persistent staging, packed model projections, a captured greedy epilogue, and batch-oriented asynchronous token transfer; validate each stage independently before proceeding.

**Tech Stack:** Python 3.11, PyTorch/torch-npu ACL Graph APIs, NumPy pinned staging, pytest, Ascend 910B1, existing auto-infer benchmark and NPU verification scripts.

## Global Constraints

- Primary model is `/data1/models/Qwen3-0.6B`, BF16, maximum model length 2048.
- Primary performance gates are B1 TPOT and B16 throughput for 128 generated tokens with EOS ignored.
- Use at least one warm-up and three measured samples; retain raw samples and compare medians.
- Every production behavior change follows a witnessed red-green TDD cycle.
- Run correctness before performance at every stage.
- Do not change public request, serving, `BatchPlan`, or `ExecutionResult` semantics.
- Do not make a lower-precision logits path default without token and numerical parity evidence.
- Preserve eager fallback, general sampling, TP, SP, EP, cancellation, and preemption behavior.
- Do not retain a performance-negative optimization as the default.

---

## File structure

- Create `auto_infer/worker/graph_task_pipeline.py`: graph-task entry ownership, external events, update stream, and double-buffered dynamic metadata.
- Create `auto_infer/worker/decode_input_stager.py`: fixed-address gear inputs, pinned staging, dirty-row block-table updates.
- Create `auto_infer/worker/decode_epilogue.py`: host-known greedy eligibility and captured/general epilogue selection.
- Modify `auto_infer/layers/attention/gqa.py`: capture event-aware GQA graph tasks and accept an explicit update stream.
- Modify `auto_infer/layers/attention/mla.py`: apply the same graph-task contract to MLA.
- Modify `auto_infer/worker/graph_decode_runner.py`: compose the three new units and capture the greedy epilogue.
- Modify `auto_infer/layers/sampler.py`: exact all-greedy fast path.
- Modify `auto_infer/layers/sampling_meta.py`: expose host-known fast-path eligibility.
- Modify `auto_infer/models/base.py`: model-owned packed projection preparation and logits precision policy.
- Modify `auto_infer/models/qwen2.py`: packed QKV/gate-up execution for Qwen2/Qwen3 and TP-local packing.
- Modify `auto_infer/engine/execution.py`: immutable `DeviceTokenBatch` and row references.
- Modify `auto_infer/engine/executor.py`: batch-oriented sampled-token protocol.
- Modify `auto_infer/engine/engine_core.py`: retain device token batches instead of cloned scalar tensors.
- Modify `auto_infer/worker/model_runner.py`: vectorized token splice and event-backed D2H for eager paged execution.
- Modify `benchmarks/run_auto_infer.py`: report optimization path counters and allow matched sync/async selection.
- Add/modify focused tests under `tests/` and NPU verification scripts under `scripts/` as listed per task.

---

### Task 1: Pipeline graph-task updates off the replay critical path

**Files:**
- Create: `auto_infer/worker/graph_task_pipeline.py`
- Modify: `auto_infer/layers/attention/gqa.py`
- Modify: `auto_infer/layers/attention/mla.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Test: `tests/test_graph_task_pipeline.py`
- Test: `tests/test_graph_decode_runner.py`
- Create: `scripts/verify_graph_task_pipeline.py`

**Interfaces:**
- Produces: `GraphTaskEntry(handle, inputs, outputs, event)`.
- Produces: `GraphTaskPipeline(entries, update_stream, metadata_slots)` with `replay(graph, ctx) -> None`.
- Changes backend `update(ctx, stream=None) -> None`; omitting `stream` preserves eager diagnostic behavior.
- `_Gear.pipeline` owns one pipeline and two independent `list[int]` KV-length slots.

- [ ] **Step 1: Write failing host ordering and ownership tests**

```python
def test_pipeline_enqueues_replay_before_side_stream_updates():
    trace = []
    pipeline = GraphTaskPipeline.for_testing(trace, num_metadata_slots=2)
    pipeline.replay(FakeGraph(trace), FakeContext([7, 11]))
    assert trace == ["replay", "update:side", "record-events:side"]


def test_pipeline_rotates_without_mutating_inflight_metadata():
    pipeline = GraphTaskPipeline.for_testing([], num_metadata_slots=2)
    first = pipeline.stage_kv_lengths([7, 11])
    second = pipeline.stage_kv_lengths([8, 12])
    assert first == [7, 11]
    assert second == [8, 12]
    assert first is not second
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_graph_task_pipeline.py tests/test_graph_decode_runner.py`

Expected: collection fails because `GraphTaskPipeline` and `_Gear.pipeline` do not exist.

- [ ] **Step 3: Implement event-aware capture and side-stream update**

During every graph attention capture, create an `ExternalEvent`, insert
`event.wait(current_stream)` and `event.reset(current_stream)` immediately before
`graph_task_group_begin`, and store the event with the task entry. Implement:

```python
class GraphTaskPipeline:
    def replay(self, graph, ctx):
        staged_ctx = self._stage_context(ctx)
        graph.replay()
        with torch.npu.stream(self.update_stream):
            self.backend.update(staged_ctx, stream=self.update_stream)
```

Each backend records the entry event on the update stream after
`graph_task_update_end`. Prime the captured graph once before serving requests.

- [ ] **Step 4: Run GREEN host tests and complete regression suite**

Run: `pytest -q tests/test_graph_task_pipeline.py tests/test_graph_decode_runner.py`

Expected: all selected tests pass.

Run: `pytest -q`

Expected: complete host suite passes.

- [ ] **Step 5: Verify NPU ordering and parity**

Run on NPU2:

```bash
python scripts/verify_graph_task_pipeline.py /data1/models/Qwen3-0.6B
python scripts/smoke_graph_engine.py /data1/models/Qwen3-0.6B
python scripts/verify_deepseek_graphdecode.py /data1/models/deepseek-ai/DeepSeek-V2-Lite
```

Expected: first replay, 200 changing-length replays, alternating gears 1/16/4/16,
Qwen3 greedy parity, and DeepSeek graph parity pass without deadlock.

- [ ] **Step 6: Benchmark and retain only a positive result**

Run the comparison manifest before and after the task. Expected primary result:
the approximately 4.8 ms serialized update phase is absent from B16's critical
path, while tokens remain identical.

- [ ] **Step 7: Commit Task 1**

```bash
git add auto_infer/worker/graph_task_pipeline.py auto_infer/layers/attention/gqa.py auto_infer/layers/attention/mla.py auto_infer/worker/graph_decode_runner.py tests/test_graph_task_pipeline.py tests/test_graph_decode_runner.py scripts/verify_graph_task_pipeline.py
git commit -m "perf: pipeline graph task updates"
```

---

### Task 2: Add an exact all-greedy sampling fast path

**Files:**
- Modify: `auto_infer/layers/sampler.py`
- Modify: `auto_infer/layers/sampling_meta.py`
- Test: `tests/test_sampler.py`
- Test: `tests/test_sampling_meta.py`

**Interfaces:**
- Produces: `SamplingTensors.all_greedy_unprocessed: bool`.
- `sample_batched(logits, tensors, generator=None)` remains source compatible.

- [ ] **Step 1: Write a failing test that forbids stochastic ops**

```python
def test_all_greedy_unprocessed_skips_softmax_and_multinomial(monkeypatch):
    tensors = _greedy_tensors(2, 3)
    tensors.all_greedy_unprocessed = True
    monkeypatch.setattr(torch, "softmax", lambda *a, **k: pytest.fail("softmax called"))
    monkeypatch.setattr(torch, "multinomial", lambda *a, **k: pytest.fail("multinomial called"))
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 2.0]])
    assert sample_batched(logits, tensors).tolist() == [1, 0]
```

Add metadata tests proving bias, disallowed tokens, penalties, or any positive
temperature set the flag to `False`.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_sampler.py tests/test_sampling_meta.py`

Expected: failure because `all_greedy_unprocessed` is missing or softmax is called.

- [ ] **Step 3: Implement the minimal fast path**

At the beginning of `sample_batched`, after processor application eligibility is
known, return `logits.argmax(dim=-1)` when `all_greedy_unprocessed` is true.
Keep the existing heterogeneous path byte-for-byte equivalent otherwise.

- [ ] **Step 4: Verify sampler behavior**

Run: `pytest -q tests/test_sampler.py tests/test_sampling_meta.py`

Expected: all tests pass, including seeded top-k/top-p cases.

- [ ] **Step 5: Run NPU token parity and paired benchmark**

Expected: greedy output digest matches the pre-change auto-infer digest and B16
throughput improves relative to Task 1's median.

- [ ] **Step 6: Commit Task 2**

```bash
git add auto_infer/layers/sampler.py auto_infer/layers/sampling_meta.py tests/test_sampler.py tests/test_sampling_meta.py
git commit -m "perf: bypass stochastic work for greedy batches"
```

---

### Task 3: Replace per-step tensor construction with persistent staging

**Files:**
- Create: `auto_infer/worker/decode_input_stager.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/model_runner.py`
- Test: `tests/test_decode_input_stager.py`
- Modify: `tests/test_input_buffers.py`
- Modify: `tests/test_graph_decode_runner.py`

**Interfaces:**
- Produces: `DecodeInputStager.stage(plan, gear_size, scratch0) -> StagedDecodeInput`.
- `StagedDecodeInput` contains stable device views plus `kv_lengths` and `order`.
- Produces counters `copied_block_rows` and `copied_block_elements` for tests and profiling.

- [ ] **Step 1: Write failing persistence and dirty-row tests**

```python
def test_stager_reuses_device_addresses_across_steps():
    stager = cpu_stager(gear=4, max_blocks=8)
    first = stager.stage(plan_a, scratch0=100)
    addresses = first.data_ptrs()
    second = stager.stage(plan_b, scratch0=100)
    assert second.data_ptrs() == addresses


def test_unchanged_block_rows_are_not_copied_twice():
    stager = cpu_stager(gear=4, max_blocks=8)
    stager.stage(plan_a, scratch0=100)
    copied = stager.copied_block_rows
    stager.stage(plan_a, scratch0=100)
    assert stager.copied_block_rows == copied
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_decode_input_stager.py tests/test_input_buffers.py tests/test_graph_decode_runner.py`

Expected: failure because `DecodeInputStager` does not exist.

- [ ] **Step 3: Implement persistent pinned/device staging**

Allocate host tensors with `pin_memory()` when available and expose NumPy views.
Allocate fixed device tensors once per gear. Fill tokens, positions, slots, and
KV lengths by slice; compare block-table rows against a host shadow; issue
non-blocking copies only for dirty contiguous row spans. Preserve scratch rows.

- [ ] **Step 4: Integrate graph and eager runners**

Replace `torch.tensor(..., device=dev)` in `_graph_submit`. Reuse the same staging
implementation in `NpuModelRunner`, retaining grow-only capacity for eager mode.

- [ ] **Step 5: Verify host and NPU behavior**

Run: `pytest -q tests/test_decode_input_stager.py tests/test_input_buffers.py tests/test_graph_decode_runner.py`

Run the full host suite and Qwen3 paged/graph/preemption scripts on NPU2.
Expected: stable addresses, no stale rows after gear shrink/grow, and unchanged tokens.

- [ ] **Step 6: Benchmark input-copy phase**

Expected: B16 isolated copy/staging time is lower than the prior approximately
1.02 ms measurement and no steady-state device allocation appears in profiler output.

- [ ] **Step 7: Commit Task 3**

```bash
git add auto_infer/worker/decode_input_stager.py auto_infer/worker/graph_decode_runner.py auto_infer/worker/model_runner.py tests/test_decode_input_stager.py tests/test_input_buffers.py tests/test_graph_decode_runner.py
git commit -m "perf: persist decode input staging buffers"
```

---

### Task 4: Pack QKV and gate/up projections

**Files:**
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/models/qwen2.py`
- Modify: `auto_infer/layers/attention/gqa.py`
- Modify: `auto_infer/layers/attention/base.py`
- Test: `tests/test_qwen2_forward.py`
- Test: `tests/test_weight_loader.py`
- Test: `tests/test_attention_backend.py`
- Modify: `scripts/smoke_qwen3.py`

**Interfaces:**
- Produces: `BaseCausalLM.prepare_packed_projections() -> None`, idempotent.
- Produces packed keys `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight`.
- Produces helper `_split_qkv(projected, q_size, kv_size)` returning views.
- Quantized packed values remain `(int8_weight_transposed, output_scale)` tuples.

- [ ] **Step 1: Write failing packed-projection parity tests**

```python
def test_packed_qkv_matches_three_independent_linears():
    packed = torch.cat([qw, kw, vw], dim=0)
    qkv = _lin(x, packed)
    q, k, v = _split_qkv(qkv, qw.shape[0], kw.shape[0])
    torch.testing.assert_close(q, x @ qw.t())
    torch.testing.assert_close(k, x @ kw.t())
    torch.testing.assert_close(v, x @ vw.t())


def test_prepare_packed_projections_is_idempotent_and_releases_sources():
    model.prepare_packed_projections()
    model.prepare_packed_projections()
    assert "model.layers.0.self_attn.qkv_proj.weight" in model.w
    assert "model.layers.0.self_attn.q_proj.weight" not in model.w
```

Add BF16, bias, TP-local, and W8A8 tuple cases.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_qwen2_forward.py tests/test_weight_loader.py tests/test_attention_backend.py`

Expected: failure because packing APIs and packed execution are missing.

- [ ] **Step 3: Implement load-time packing after TP sharding**

Concatenate output rows for floating weights. For W8A8 concatenate transposed
input-major int8 weights on the output dimension and concatenate per-output
scales. Delete replaced source keys after all shared paths use the packed keys.

- [ ] **Step 4: Route every Qwen-family forward path through packed helpers**

Update dense, eager paged, graph, and context-parallel paths. Use one QKV `_lin`
and one gate/up `_lin` per layer. Preserve q/k normalization, RoPE, TP reduction,
and optional projection bias.

- [ ] **Step 5: Verify host/NPU/TP parity**

Run focused tests, full host suite, Qwen3 dense/HF first-token parity, paged
prefill parity, graph parity, and TP2 smoke. Expected: unchanged greedy tokens
and no duplicate unpacked projection weights in the loaded model.

- [ ] **Step 6: Benchmark replay**

Expected: graph replay time and kernel count decrease; retain raw profiler output.

- [ ] **Step 7: Commit Task 4**

```bash
git add auto_infer/models/base.py auto_infer/models/qwen2.py auto_infer/layers/attention/gqa.py auto_infer/layers/attention/base.py tests/test_qwen2_forward.py tests/test_weight_loader.py tests/test_attention_backend.py scripts/smoke_qwen3.py
git commit -m "perf: pack qkv and gate-up projections"
```

---

### Task 5: Capture logits and greedy argmax in the decode graph

**Files:**
- Create: `auto_infer/worker/decode_epilogue.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/models/base.py`
- Test: `tests/test_decode_epilogue.py`
- Modify: `tests/test_graph_decode_runner.py`
- Create: `scripts/verify_greedy_epilogue_graph.py`

**Interfaces:**
- Produces: `DecodeEpilogue.is_capturable_greedy(requests) -> bool`.
- `_Gear.logits` is a fixed `(gear, vocab)` tensor and `_Gear.sampled` is a fixed `(gear,)` tensor.
- Graph handles expose both hidden output and captured greedy-token output.

- [ ] **Step 1: Write failing eligibility and output-buffer tests**

```python
def test_only_unprocessed_greedy_batches_use_captured_epilogue():
    assert DecodeEpilogue.is_capturable_greedy([plain_greedy_request()])
    assert not DecodeEpilogue.is_capturable_greedy([temperature_request(0.8)])
    assert not DecodeEpilogue.is_capturable_greedy([biased_greedy_request()])


def test_gear_owns_fixed_logits_and_token_outputs():
    gear = cpu_gear(4, hidden=8, vocab=32)
    assert gear.logits.shape == (4, 32)
    assert gear.sampled.shape == (4,)
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_decode_epilogue.py tests/test_graph_decode_runner.py`

Expected: missing epilogue and output buffers.

- [ ] **Step 3: Capture forward, logits, and argmax**

Within the existing graph capture, copy logits into the gear buffer and argmax
into the sampled-token buffer. For capturable greedy requests return a view of
that buffer without launching external logits or sampler operations. For other
requests consume the captured logits buffer in the general sampler.

- [ ] **Step 4: Gate logits precision with evidence**

Measure FP32, BF16, and mixed accumulation against FP32 logits. Enable the
fastest candidate only if all tested prompt/token streams match and numerical
error stays within the documented tolerance; otherwise keep FP32 in graph.

- [ ] **Step 5: Verify graph epilogue correctness**

Run host tests, then `scripts/verify_greedy_epilogue_graph.py` on NPU2 for B1,
B16, alternating gears, 256 decode steps, and external-sampler fallback.

- [ ] **Step 6: Benchmark full decode epilogue**

Expected: no per-step external lm-head or greedy sampling launch for the primary
workload; report memory increase from fixed logits buffers separately.

- [ ] **Step 7: Commit Task 5**

```bash
git add auto_infer/worker/decode_epilogue.py auto_infer/worker/graph_decode_runner.py auto_infer/models/base.py tests/test_decode_epilogue.py tests/test_graph_decode_runner.py scripts/verify_greedy_epilogue_graph.py
git commit -m "perf: capture greedy decode epilogue"
```

---

### Task 6: Make async token handoff batch-oriented and event-backed

**Files:**
- Modify: `auto_infer/engine/execution.py`
- Modify: `auto_infer/engine/executor.py`
- Modify: `auto_infer/engine/engine_core.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Test: `tests/test_device_token_batch.py`
- Modify: `tests/test_async_output_thread.py`
- Modify: `tests/test_engine_core.py`
- Modify: `scripts/verify_async_sched.py`
- Modify: `scripts/bench_async_scheduling.py`

**Interfaces:**
- Produces immutable `DeviceTokenBatch(tokens, order, row_by_request)`.
- `Executor.sampled_of(handle) -> DeviceTokenBatch | None` replaces the scalar dictionary internally.
- `DecodeInputStager.splice(batch, request_order) -> None` performs one vectorized gather/copy.
- `AsyncHostCopy(tokens, pinned_cpu, ready_event)` owns asynchronous D2H completion.

- [ ] **Step 1: Write failing no-clone/no-scalar-splice tests**

```python
def test_sampled_batch_retains_one_tensor_without_per_row_clones():
    tokens = torch.tensor([3, 5, 7])
    batch = DeviceTokenBatch.from_output(tokens, ["a", "b", "c"])
    assert batch.tokens.data_ptr() == tokens.data_ptr()
    assert batch.rows_for(["c", "a"]) == [2, 0]


def test_async_splice_is_one_vectorized_operation():
    trace = []
    stager = tracing_stager(trace)
    stager.splice(device_batch(), ["b", "a"])
    assert trace == ["index_select", "copy"]
```

Add queue-drain, cancellation, preemption, skipped-request, EOS-in-lookahead,
and sync/async token-identity cases.

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_device_token_batch.py tests/test_async_output_thread.py tests/test_engine_core.py`

Expected: missing batch token API and existing scalar dictionary behavior fails assertions.

- [ ] **Step 3: Replace scalar token ownership**

Retain complete sampled tensors in `DeviceTokenBatch`; replace `_sampled` scalar
entries with batch row references whose owners stay alive until overwritten or
freed. Group rows by owner and perform vectorized gathers into the input buffer.

- [ ] **Step 4: Implement event-backed asynchronous D2H**

Copy sampled tokens into a reusable pinned CPU buffer on a dedicated copy stream,
record a ready event, and give the output thread an `AsyncHostCopy`. The worker
waits for the event and maps already-host-resident values to request IDs. Remove
device `.tolist()` from the worker thread.

- [ ] **Step 5: Verify correctness and lifecycle**

Run focused and full host suites. On NPU2 run sync/async parity, 48 concurrent
requests, cancellation followed by 60 requests, forced preemption, and service
shutdown with pending copies. Expected: identical tokens, no stale references,
no deadlock, and clean thread shutdown.

- [ ] **Step 6: Benchmark scheduler modes**

Measure depth 1/2/3 at B1/B16. Async may become default only if it is neutral or
faster at both acceptance points; otherwise retain automatic sync selection for
the primary workload and report the measured boundary.

- [ ] **Step 7: Commit Task 6**

```bash
git add auto_infer/engine/execution.py auto_infer/engine/executor.py auto_infer/engine/engine_core.py auto_infer/worker/model_runner.py auto_infer/worker/graph_decode_runner.py tests/test_device_token_batch.py tests/test_async_output_thread.py tests/test_engine_core.py scripts/verify_async_sched.py scripts/bench_async_scheduling.py
git commit -m "perf: batch asynchronous token handoff"
```

---

### Task 7: Final correctness, stability, and three-framework gate

**Files:**
- Modify: `benchmarks/run_auto_infer.py`
- Modify: `docs/ARCHITECTURE-COMPARISON.md`
- Create: `docs/DECODE-PERFORMANCE-VALIDATION-2026-07-19.md`
- Test: `tests/test_benchmark_manifest.py`
- Test: `tests/test_documentation_integrity.py`

**Interfaces:**
- Benchmark result adds `path_counters`, `async_mode`, and raw phase samples without removing existing fields.

- [ ] **Step 1: Write failing benchmark-schema tests**

Assert the comparison result records optimization path counters, scheduler mode,
raw samples, model, dtype, batch, output length, memory, and output digest.

- [ ] **Step 2: Run RED schema tests**

Run: `pytest -q tests/test_benchmark_manifest.py tests/test_documentation_integrity.py`

Expected: missing result fields and validation report.

- [ ] **Step 3: Implement reporting and validation document**

Record every command, version, device occupancy check, raw sample, median,
correctness result, memory figure, and output digest. Separate startup, TTFT,
TPOT, and steady throughput.

- [ ] **Step 4: Run complete local verification**

Run:

```bash
pytest -q
python -m compileall -q auto_infer benchmarks scripts tests
git diff --check
```

Expected: zero failures and zero whitespace errors.

- [ ] **Step 5: Run complete NPU2 correctness/stability matrix**

Run the existing dense, paged, graph, async, IPC, concurrent serving,
cancellation/stability, TP2, SP2/EP2, parallel-mesh, and Moonlight validation
commands recorded in `docs/PHASE1-VALIDATION-2026-07-19.md`, plus the three new
NPU verification scripts from this plan.

- [ ] **Step 6: Run matched three-framework comparison**

Recheck `npu-smi`, run auto-infer, Omni-NPU, and vLLM-Ascend sequentially with
the same manifest, and collect at least three measured samples per metric.

Expected final gate: auto-infer B16 median throughput is at least Omni-NPU's and
auto-infer B1 median TPOT is no higher than Omni-NPU's. If either misses, keep
the goal open and use the new phase counters/profiler to identify the remaining
critical path instead of declaring completion.

- [ ] **Step 7: Commit Task 7**

```bash
git add benchmarks/run_auto_infer.py docs/ARCHITECTURE-COMPARISON.md docs/DECODE-PERFORMANCE-VALIDATION-2026-07-19.md tests/test_benchmark_manifest.py tests/test_documentation_integrity.py
git commit -m "docs: validate decode performance convergence"
```
