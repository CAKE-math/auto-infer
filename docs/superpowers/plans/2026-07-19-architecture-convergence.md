# Auto-Infer Architecture Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge auto-infer on one validated configuration, execution contract, serving lifecycle, IPC protocol, and parallel mesh, then verify correctness and stability on npu2 under `/data2`.

**Architecture:** EngineCore owns logical state and exchanges immutable `BatchPlan`/`ExecutionResult` values with a common executor. A persistent `EngineService` plus `RequestBroker` supports both thread and process transports, while `ParallelMesh` provides named distributed axes and the attention registry removes backend construction from model subclasses.

**Tech Stack:** Python 3.11+, dataclasses, typing protocols, asyncio/threading/multiprocessing, PyTorch/torch_npu, pytest, HCCL, rsync/SSH.

## Global Constraints

- Internal APIs may break; preserve `LLM.generate` and OpenAI HTTP semantics.
- `EngineConfig` is the only runtime configuration source.
- Every behavior change follows red-green-refactor.
- Do not overwrite existing `/data2/auto-infer-eval*` directories.
- Do not select an NPU already in use.
- Phase one is accepted on architecture, correctness, and stability before the three-framework comparison.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Fail-fast Configuration and Progress Guarantees

**Files:**
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/engine/request.py`
- Modify: `auto_infer/engine/scheduler.py`
- Modify: `auto_infer/engine/engine_core.py`
- Modify: `auto_infer/entrypoints/llm.py`
- Test: `tests/test_config.py`
- Test: `tests/test_scheduler.py`
- Test: `tests/test_engine_core.py`

**Interfaces:**
- Produces: `ConfigurationError`, `RequestRejectedError`, `EngineStalledError`.
- Produces: `EngineConfig.validate_runtime(world_size: int | None = None) -> None`.
- Produces: `Scheduler.reject_unschedulable() -> list[tuple[Request, Exception]]`.

- [ ] **Step 1: Write failing tests for invalid limits, duplicate ids, oversized prompts, and no-progress detection**

```python
def test_oversized_prompt_fails_instead_of_spinning():
    llm = _llm(num_blocks=1, block_size=4)
    with pytest.raises(RequestRejectedError, match="requires 2 KV blocks"):
        llm.generate([[1, 2, 3, 4, 5]], max_tokens=1)

def test_duplicate_request_id_is_rejected():
    scheduler.add_request(Request("same", [1], SamplingParams()))
    with pytest.raises(RequestRejectedError, match="duplicate"):
        scheduler.add_request(Request("same", [2], SamplingParams()))
```

- [ ] **Step 2: Run focused tests and verify the expected failures**

Run: `pytest -q tests/test_config.py tests/test_scheduler.py tests/test_engine_core.py`

Expected: failures because the exception types and validation behavior do not exist.

- [ ] **Step 3: Implement validation, explicit request failure, and per-step progress accounting**

```python
class RequestRejectedError(ValueError):
    pass

class EngineStalledError(RuntimeError):
    pass

def add_request(self, req: Request) -> None:
    if req.request_id in self._requests:
        raise RequestRejectedError(f"duplicate request id: {req.request_id}")
    required = self.kv.blocks_needed(req.num_prefill_tokens)
    if required > self.kv.num_blocks:
        raise RequestRejectedError(
            f"request {req.request_id} requires {required} KV blocks; capacity is {self.kv.num_blocks}")
```

- [ ] **Step 4: Run focused and full host tests**

Run: `pytest -q tests/test_config.py tests/test_scheduler.py tests/test_engine_core.py && pytest -q`

Expected: all tests pass and oversized requests terminate immediately.

### Task 2: Immutable Execution Contract

**Files:**
- Create: `auto_infer/engine/execution.py`
- Modify: `auto_infer/engine/executor.py`
- Modify: `auto_infer/engine/engine_core.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Test: `tests/test_execution_contract.py`
- Modify: existing engine/runner tests

**Interfaces:**
- Produces: frozen `RequestView`, `BatchPlan`, `ExecutionResult`, `ExecutionStats`.
- Produces: `BatchPlan.from_scheduler(output, scheduler) -> BatchPlan`.
- Changes: `Executor.execute(plan: BatchPlan) -> ExecutionResult`.
- Changes: `Executor.submit(plan: BatchPlan, previous: Mapping[str, Any]) -> Any`.

- [ ] **Step 1: Write failing immutability and mock parity tests**

```python
def test_request_view_is_immutable_and_detached():
    plan = BatchPlan.from_scheduler(output, scheduler)
    with pytest.raises(FrozenInstanceError):
        plan.requests[0].num_computed_tokens = 9
    scheduler.get_request("r").output_token_ids.append(99)
    assert plan.requests[0].output_token_ids == ()

def test_sync_and_async_mock_execution_results_match():
    assert sync_result.tokens == async_result.tokens
```

- [ ] **Step 2: Verify failures before introducing execution dataclasses**

Run: `pytest -q tests/test_execution_contract.py`

Expected: import or assertion failures for missing execution types.

- [ ] **Step 3: Add frozen execution values and migrate MockExecutor/EngineCore**

```python
@dataclass(frozen=True)
class ExecutionResult:
    tokens: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    next_drafts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)
    stats: ExecutionStats = field(default_factory=ExecutionStats)
```

- [ ] **Step 4: Migrate NPU runners without compatibility access to Scheduler**

Runner input marshaling reads only `RequestView` and scheduled metadata. EngineCore alone mutates `Request` and KV lifecycle state.

- [ ] **Step 5: Run contract tests and the full host suite**

Run: `pytest -q tests/test_execution_contract.py tests/test_engine_core.py tests/test_graph_decode_runner.py tests/test_spec_decode.py && pytest -q`

Expected: all execution modes satisfy one result structure.

### Task 3: Common Executor Construction and Lifecycle

**Files:**
- Create: `auto_infer/engine/factory.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/engine/executor.py`
- Modify: `auto_infer/engine/npu_executor.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Modify: `auto_infer/entrypoints/llm.py`
- Modify: `auto_infer/serving/api_server.py`
- Modify: `auto_infer/eval/runner.py`
- Test: `tests/test_executor_factory.py`

**Interfaces:**
- Produces: `ExecutionConfig(mode, device_index, graph_max_gear, force_eager)`.
- Produces: `build_executor(config: EngineConfig) -> Executor`.
- Produces: `Executor.close()`, `Executor.recoverable`.

- [ ] **Step 1: Write failing tests proving one config reaches model and runner construction**

```python
def test_factory_propagates_all_capacity_and_dtype_fields(monkeypatch):
    executor = build_executor(config)
    assert executor.runner.capacity == RunnerCapacity(
        num_blocks=32, block_size=8, max_num_seqs=7,
        max_num_batched_tokens=99, max_model_len=128)
```

- [ ] **Step 2: Verify factory tests fail for the missing API**

Run: `pytest -q tests/test_executor_factory.py`

- [ ] **Step 3: Implement shared model loading, runner selection, and close semantics**

All public entrypoints call `build_executor(config)` unless an explicit test executor is injected. `LLM` raises if neither an executor nor a production execution configuration is provided; mock behavior is selected by `LLM.for_testing()`.

- [ ] **Step 4: Remove duplicated config/model-loading constructors and run tests**

Run: `pytest -q tests/test_executor_factory.py tests/test_smoke.py tests/test_serving_engine.py && pytest -q`

### Task 4: Request Broker and Unified Service Lifecycle

**Files:**
- Create: `auto_infer/serving/broker.py`
- Create: `auto_infer/serving/service.py`
- Modify: `auto_infer/serving/async_engine.py`
- Modify: `auto_infer/serving/api_server.py`
- Test: `tests/test_request_broker.py`
- Modify: `tests/test_serving_engine.py`

**Interfaces:**
- Produces: `RequestBroker.submit`, `emit`, `finish`, `fail`, `cancel`, `close`.
- Produces: `EngineService.start()`, `submit()`, `cancel()`, `close()`.
- Async and synchronous facades delegate to the same service.

- [ ] **Step 1: Write failing cancellation-race, close, and executor-failure tests**

```python
def test_cancel_during_emit_does_not_kill_service():
    first = service.submit([1, 2], SamplingParams(max_tokens=100))
    service.cancel(first.request_id)
    second = service.submit([10], SamplingParams(max_tokens=2))
    assert list(second) == [11, 12]

def test_close_joins_background_thread():
    service.close()
    assert not service.thread.is_alive()
```

- [ ] **Step 2: Verify the race/lifecycle tests fail**

Run: `pytest -q tests/test_request_broker.py tests/test_serving_engine.py`

- [ ] **Step 3: Implement broker ownership and explicit lifecycle**

The service thread is the only EngineCore owner. Broker queue removal and emission are atomic under its lock. Errors arrive as error envelopes rather than normal end sentinels.

- [ ] **Step 4: Replace duplicate AsyncEngine/ServingEngine loops with facades**

- [ ] **Step 5: Run serving and full tests**

Run: `pytest -q tests/test_request_broker.py tests/test_serving_engine.py tests/test_async_output_thread.py && pytest -q`

### Task 5: Persistent Concurrent IPC and Bounded Router

**Files:**
- Rewrite: `auto_infer/serving/ipc.py`
- Rewrite: `auto_infer/serving/router.py`
- Create: `tests/test_serving_ipc.py`
- Create: `tests/test_router.py`
- Modify: `scripts/verify_ipc_serving.py`
- Modify: `scripts/verify_router.py`

**Interfaces:**
- Produces: `RequestEnvelope(request_id, kind, payload)` and `ResponseEnvelope`.
- `EngineProcess` owns one worker-side `EngineService` and one client-side demux thread.
- `Router` decrements live load and bounds affinity with configurable LRU capacity.

- [ ] **Step 1: Write failing multi-client cross-routing and persistent-engine tests**

```python
def test_concurrent_streams_are_demultiplexed_by_request_id():
    outputs = run_concurrently(
        lambda: list(proc.generate_stream("a", [1], 3)),
        lambda: list(proc.generate_stream("b", [10], 3)))
    assert outputs == [[2, 3, 4], [11, 12, 13]]

def test_router_live_load_returns_to_zero():
    list(router.generate_stream("r", [1], 1))
    assert router.live_load == (0, 0)
```

- [ ] **Step 2: Verify IPC/router tests fail against shared queue consumption**

Run: `pytest -q tests/test_serving_ipc.py tests/test_router.py`

- [ ] **Step 3: Implement envelopes, persistent worker service, demux, cancellation, and shutdown**

- [ ] **Step 4: Implement bounded affinity and live-load accounting**

- [ ] **Step 5: Run IPC/router and full tests**

Run: `pytest -q tests/test_serving_ipc.py tests/test_router.py && pytest -q`

### Task 6: Validated Named-Axis Parallel Mesh

**Files:**
- Create: `auto_infer/distributed/mesh.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/distributed/parallel_state.py`
- Modify: `tests/test_sp_ep_mesh.py`
- Modify: `tests/test_two_level_groups.py`
- Create: `tests/test_parallel_mesh.py`
- Modify: NPU distributed verification scripts

**Interfaces:**
- Produces: frozen `ParallelMesh(tp, dp, ep, cp, sp, nnodes)`.
- Produces: `coordinate(rank)`, `groups(axis)`, `validate(world_size)`.
- Collectives resolve a named axis group.

- [ ] **Step 1: Write failing property tests for coverage and invalid products**

```python
@pytest.mark.parametrize("axis", ["tp", "dp", "ep", "cp", "sp"])
def test_axis_groups_cover_each_rank_once(mesh, axis):
    groups = mesh.groups(axis)
    assert sorted(r for group in groups for r in group) == list(range(mesh.world_size))

def test_world_size_mismatch_is_rejected():
    with pytest.raises(ConfigurationError, match="WORLD_SIZE"):
        ParallelMesh(tp=2, dp=2).validate(2)
```

- [ ] **Step 2: Verify mesh tests fail before implementation**

Run: `pytest -q tests/test_parallel_mesh.py`

- [ ] **Step 3: Implement deterministic named-axis groups and migrate initialization**

- [ ] **Step 4: Remove implicit CP=TP and DP=WORLD/TP behavior**

- [ ] **Step 5: Run all distributed host tests**

Run: `pytest -q tests/test_parallel_mesh.py tests/test_sp_ep_mesh.py tests/test_two_level_groups.py`

### Task 7: Attention Backend Registry and Model Contract

**Files:**
- Create: `auto_infer/layers/attention/registry.py`
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/models/qwen2.py`
- Modify: `auto_infer/models/deepseek_v2.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: graph runners
- Create: `tests/test_extension_contracts.py`
- Modify: attention/model tests

**Interfaces:**
- Produces: `AttentionSpec(family, parameters)` from model metadata.
- Produces: `build_attention_backend(spec, mode, capacity, device, dtype, weights)`.
- Model subclasses no longer expose `make_dense_backend`, `make_attention_backend`, or `make_graph_backend`.

- [ ] **Step 1: Write failing extension-independence tests**

```python
def test_existing_model_selects_new_registered_backend_without_model_change():
    register_backend("fake", "paged", FakeBackend)
    backend = build_attention_backend(
        fake_spec,
        "paged",
        RunnerCapacity(num_blocks=4, block_size=16),
        torch.device("cpu"),
        torch.float32,
        {},
    )
    assert isinstance(backend, FakeBackend)

def test_model_contract_has_no_runtime_backend_factories():
    assert not hasattr(BaseCausalLM, "make_graph_backend")
```

- [ ] **Step 2: Verify extension tests fail against model-owned factories**

Run: `pytest -q tests/test_extension_contracts.py`

- [ ] **Step 3: Implement metadata and backend registry, then migrate GQA/MLA**

- [ ] **Step 4: Remove factories and run model/attention/graph tests**

Run: `pytest -q tests/test_extension_contracts.py tests/test_attention_backend.py tests/test_qwen2_forward.py tests/test_deepseek_v2_forward.py tests/test_graph_decode_runner.py && pytest -q`

### Task 8: Public API, CLI, and Evidence Documentation

**Files:**
- Modify: `auto_infer/__init__.py`
- Modify: `pyproject.toml`
- Create: `auto_infer/entrypoints/cli.py`
- Rewrite: `README.md`
- Rewrite: `docs/SPEC-ALIGNMENT.md`
- Create: `tests/test_documentation_integrity.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Exports supported configuration, LLM, request errors, and factory types.
- Adds `auto-infer` console script.
- Capability evidence contains only paths verified by the integrity test.

- [ ] **Step 1: Write failing public-import, CLI help, and documentation-link tests**

```python
def test_capability_paths_exist():
    for path in documented_evidence_paths():
        assert path.exists(), path

def test_public_api_exports_llm_and_config():
    assert auto_infer.LLM is LLM
    assert auto_infer.EngineConfig is EngineConfig
```

- [ ] **Step 2: Verify tests expose stale references and missing exports**

Run: `pytest -q tests/test_documentation_integrity.py tests/test_smoke.py`

- [ ] **Step 3: Add exports/CLI and replace unsupported documentation claims**

- [ ] **Step 4: Run the complete local verification gate**

Run: `pytest -q && python -m compileall -q auto_infer && git diff --check`

Expected: all tests pass, compilation succeeds, and no whitespace errors exist.

### Task 9: Isolated npu2 Phase-one Validation

**Files:**
- Create logs under `/data2/auto-infer-architecture-20260719` on npu2.
- Modify local NPU verification scripts only when a failing NPU test exposes a reproducible defect; follow TDD with a host or NPU regression test.

**Interfaces:**
- Consumes the phase-one local verification result.
- Produces retained command logs and a pass/fail matrix.

- [ ] **Step 1: Check NPU availability and create an explicit isolated target directory**

Run: `ssh npu2 'npu-smi info'`

Expected: select only devices with no running process and sufficient free HBM.

- [ ] **Step 2: Rsync source without models, caches, logs, or Git metadata**

Run: `rsync -az --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' auto-infer/ npu2:/data2/auto-infer-architecture-20260719/`

- [ ] **Step 3: Run the complete host suite in the target runtime**

Run through the existing Ascend environment discovered on npu2; retain stdout/stderr in `logs/host-tests.log`.

- [ ] **Step 4: Run single-card eager/paged/graph correctness and parity**

Use an available small checkpoint under `/data2/models` or the existing tested model path. Bind one free NPU explicitly and retain one log per command.

- [ ] **Step 5: Run concurrent serving, cancellation, and persistent IPC tests on NPU**

- [ ] **Step 6: Run TP2 and EP2/SP2 HCCL validation on two free NPUs**

- [ ] **Step 7: Run Moonlight-16B MLA/MoE persistent-engine validation**

- [ ] **Step 8: Summarize phase-one results with exact commands, devices, logs, and failures**

### Task 10: Three-framework Phase-two Comparison

**Files:**
- Create: `benchmarks/architecture_comparison.py`
- Create: `docs/ARCHITECTURE-COMPARISON.md`
- Reuse isolated npu2 directories for auto-infer, omni-npu, and vllm-ascend.

**Interfaces:**
- Produces common prompts and measurement schema for TTFT, TPOT, throughput, memory, failures, and variance.
- Produces an extension-cost comparison based on concrete files/interfaces changed.

- [ ] **Step 1: Write a comparison manifest fixing model, prompts, concurrency, limits, warmup, and sample count**

- [ ] **Step 2: Validate that each framework consumes the identical manifest**

- [ ] **Step 3: Run auto-infer, omni-npu, and vllm-ascend sequentially on the same free devices**

- [ ] **Step 4: Generate the evidence-backed architecture/runtime comparison**

- [ ] **Step 5: Run final verification and report whether auto-infer exceeds both baselines**

Run: `pytest -q && git diff --check`

Expected: all local tests pass; comparison claims cite retained npu2 logs.
