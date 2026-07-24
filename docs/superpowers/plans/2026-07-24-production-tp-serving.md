# Production Tensor-Parallel Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve BF16 Qwen-family models across 2–8 NPUs through the existing
production FastAPI/AsyncEngine stack with rank-local weights and KV, no
steady-state per-step Host control collective, and replica-wide failure.

**Architecture:** Qwen supplies a validated checkpoint-name slice plan consumed
by the shared Safetensors loader. Every TP rank runs the same EngineCore; rank 0
publishes epoch-tagged control changes to follower receiver threads, while a
parent supervisor treats all ranks as one failure domain. Rank 0 attaches the
SPMD service to the unchanged AsyncEngine and ApiRuntime.

**Tech Stack:** Python 3.10+, PyTorch/torch-npu distributed HCCL, Safetensors,
multiprocessing spawn queues, FastAPI/Uvicorn, pytest.

## Global Constraints

- `tp_size == 1` must add no process, collective, or hot-loop branch.
- BF16 is the only enabled TP weight format; quantization remains a reserved interface.
- Initial TP models are dense Qwen GQA; MLA/DeepSeek TP and TP MTP fail explicitly.
- Initial topology is one node with unique physical NPU IDs.
- No `broadcast_object_list` is allowed in the steady decode loop.
- Any rank failure terminates the complete TP replica.
- Existing FastAPI, admission, SSE, metrics, prefix-cache, and AsyncEngine behavior stays shared.

---

### Task 1: Validated rank-local weight loading

**Files:**
- Create: `auto_infer/models/parallel.py`
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/models/loader.py`
- Modify: `auto_infer/models/qwen2.py`
- Modify: `auto_infer/engine/factory.py`
- Test: `tests/test_tensor_parallel_loading.py`

**Interfaces:**
- Produces: `TensorParallelPlan.for_qwen(config, rank, size)`
- Produces: `TensorParallelPlan.slice_spec(name) -> tuple[int, int, int] | None`
- Produces: `BaseCausalLM.SUPPORTS_TENSOR_PARALLEL: bool`
- Extends: `load_sharded(..., slicer=None)`
- Consumes: initialized `parallel_state.tp_rank()/tp_size()`

- [ ] **Step 1: Write failing validation and slice tests**

```python
def test_qwen_tp_plan_rejects_non_divisible_kv_heads():
    cfg = SimpleNamespace(
        num_heads=16, num_kv_heads=3, intermediate_size=64, head_dim=8)
    with pytest.raises(ValueError, match="num_kv_heads"):
        TensorParallelPlan.for_qwen(cfg, rank=0, size=2)


def test_qwen_tp_plan_slices_column_and_row_weights():
    cfg = SimpleNamespace(
        num_heads=8, num_kv_heads=4, intermediate_size=32, head_dim=8)
    plan = TensorParallelPlan.for_qwen(cfg, rank=1, size=2)
    assert plan.slice_spec(
        "model.layers.0.self_attn.q_proj.weight") == (0, 32, 32)
    assert plan.slice_spec(
        "model.layers.0.self_attn.o_proj.weight") == (1, 32, 32)
    assert plan.slice_spec("model.embed_tokens.weight") is None
```

- [ ] **Step 2: Run the focused tests and verify they fail because the contract is absent**

Run: `pytest tests/test_tensor_parallel_loading.py -q`

Expected: import failure for `auto_infer.models.parallel`.

- [ ] **Step 3: Implement the immutable Qwen TP plan**

```python
@dataclass(frozen=True)
class TensorParallelPlan:
    rank: int
    size: int
    q_rows: int
    kv_rows: int
    intermediate_rows: int

    @classmethod
    def for_qwen(cls, config, rank: int, size: int):
        if size <= 0 or not 0 <= rank < size:
            raise ValueError("TP rank must satisfy 0 <= rank < size")
        for field in ("num_heads", "num_kv_heads", "intermediate_size"):
            if getattr(config, field) % size:
                raise ValueError(f"{field} must be divisible by tp_size")
        return cls(
            rank=rank,
            size=size,
            q_rows=config.num_heads // size * config.head_dim,
            kv_rows=config.num_kv_heads // size * config.head_dim,
            intermediate_rows=config.intermediate_size // size,
        )

    def slice_spec(self, name: str) -> tuple[int, int, int] | None:
        if not name.startswith("model.layers."):
            return None
        suffixes = {
            "self_attn.q_proj.weight": (0, self.q_rows),
            "self_attn.q_proj.bias": (0, self.q_rows),
            "self_attn.k_proj.weight": (0, self.kv_rows),
            "self_attn.k_proj.bias": (0, self.kv_rows),
            "self_attn.v_proj.weight": (0, self.kv_rows),
            "self_attn.v_proj.bias": (0, self.kv_rows),
            "self_attn.o_proj.weight": (1, self.q_rows),
            "mlp.gate_proj.weight": (0, self.intermediate_rows),
            "mlp.up_proj.weight": (0, self.intermediate_rows),
            "mlp.down_proj.weight": (1, self.intermediate_rows),
        }
        for suffix, (dimension, length) in suffixes.items():
            if name.endswith(suffix):
                return dimension, self.rank * length, length
        return None
```

- [ ] **Step 4: Add a Safetensors slice-at-read parity test**

Create a temporary checkpoint with known row- and column-parallel tensors.
Load it once without `slicer` and once for every TP rank with
`plan.slice_spec`; concatenate shards and assert exact equality with the full
tensors.

- [ ] **Step 5: Verify the parity test fails because `load_sharded` has no slicer**

Run: `pytest tests/test_tensor_parallel_loading.py -q`

Expected: `TypeError: load_sharded() got an unexpected keyword argument 'slicer'`.

- [ ] **Step 6: Extend `load_sharded` and Qwen construction**

Use `safe_open.get_slice`, slice only dimension 0 or 1, make the shard
contiguous, move/cast it, then run existing QKV and gate/up packing on local
shards. Set `Qwen2Model.SUPPORTS_TENSOR_PARALLEL = True`; inherited Qwen3 and
MiMo receive the same capability.

- [ ] **Step 7: Reject unsupported classes in the composition root**

`factory.load_model` reads TP/EP state. For `tp_size > 1`, reject a model class
whose capability is false before invoking its loader; otherwise pass
`tp_rank/tp_size`. Continue passing `ep_rank/ep_size` independently.

- [ ] **Step 8: Run focused and full regression tests**

Run:

```bash
pytest tests/test_tensor_parallel_loading.py tests/test_weight_loader.py \
  tests/test_executor_factory.py -q
pytest -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add auto_infer/models/parallel.py auto_infer/models/base.py \
  auto_infer/models/loader.py auto_infer/models/qwen2.py \
  auto_infer/engine/factory.py tests/test_tensor_parallel_loading.py
git commit -m "feat: load tensor-parallel weights by rank"
```

### Task 2: Production service seams without behavior drift

**Files:**
- Modify: `auto_infer/serving/service.py`
- Modify: `auto_infer/serving/async_engine.py`
- Modify: `auto_infer/serving/api_server.py`
- Test: `tests/test_serving_engine.py`
- Test: `tests/test_api_server_architecture.py`

**Interfaces:**
- Produces: `EngineService._collect_control()`
- Produces: `EngineService._apply_control(submits, aborts)`
- Produces: `EngineService._emit_outputs(finished)`
- Produces: `AsyncEngine.from_service(service)`
- Produces: `build_runtime(..., engine)` and `run_runtime(runtime, ...)`

- [ ] **Step 1: Write failing tests for extracted seams**

Test that collecting does not mutate the EngineCore, applying preserves queued
cancellation/timing semantics, and `AsyncEngine.from_service()` uses the exact
provided service.

- [ ] **Step 2: Verify focused tests fail on absent interfaces**

Run:

```bash
pytest tests/test_serving_engine.py tests/test_api_server_architecture.py -q
```

- [ ] **Step 3: Extract control/application/output methods**

Move existing code without changing ordering:

```python
submits, aborts = self._collect_control()
self._apply_control(submits, aborts)
if self.engine.has_unfinished():
    try:
        finished = self.engine.step()
    except Exception as error:
        self._handle_step_error(error)
    else:
        self._emit_outputs(finished)
```

Keep bounded inbox, queued cancellation, timing, load snapshots, and prefix
statistics in the same owner.

- [ ] **Step 4: Add alternate service construction to AsyncEngine**

```python
@classmethod
def from_service(cls, service):
    instance = cls.__new__(cls)
    instance.service = service
    instance.engine = service.engine
    instance._active = {}
    instance._closed = False
    return instance
```

- [ ] **Step 5: Split API composition from Uvicorn execution**

`serve()` still constructs the same tokenizer/config/executor. It delegates to
shared helpers that accept an already-built async engine, construct
`ApiRuntime`, and run `create_app(runtime)` through Uvicorn.

- [ ] **Step 6: Run serving regression tests and commit**

Run:

```bash
pytest tests/test_serving_engine.py tests/test_serving_lifecycle.py \
  tests/test_text_serving_api.py tests/test_api_server_architecture.py -q
```

Commit:

```bash
git add auto_infer/serving/service.py auto_infer/serving/async_engine.py \
  auto_infer/serving/api_server.py tests
git commit -m "refactor: expose production serving composition seams"
```

### Task 3: Epoch-tagged change-only SPMD control

**Files:**
- Create: `auto_infer/serving/tp_control.py`
- Create: `auto_infer/serving/tp_service.py`
- Test: `tests/test_tp_control.py`
- Test: `tests/test_tp_service.py`

**Interfaces:**
- Produces: `ControlBatch`, `ControlAck`, `ReplicaFatal`
- Produces: `QueueControlLeader.publish(batch)`
- Produces: `QueueControlFollower.pending(epoch)`
- Produces: `SpmdEngineService`
- Consumes: the service seams from Task 2

- [ ] **Step 1: Write failing protocol tests**

Test contiguous sequence enforcement, future-epoch delivery, immediate idle
application, follower acknowledgement, shutdown, and duplicate/out-of-order
rejection.

- [ ] **Step 2: Verify tests fail because TP control types do not exist**

Run: `pytest tests/test_tp_control.py -q`

- [ ] **Step 3: Implement immutable messages and queue endpoints**

The follower owns a receiver thread. It acknowledges receipt only after storing
the next contiguous sequence. `pending(epoch)` returns messages whose
`apply_epoch <= epoch` without blocking the engine thread.

- [ ] **Step 4: Write failing two-service convergence tests**

Use paired MockExecutors and queue endpoints. Submit multiple overlapping
requests, abort one, finish the rest, and compare request IDs, scheduler state,
output tokens, prefix counters, and block tables after every applied control
epoch. Assert that a long decode with no submits/aborts does not increment the
transport publish count.

- [ ] **Step 5: Implement `SpmdEngineService`**

Rank 0 drains the production inbox, publishes a future-epoch batch while active,
and applies it at the same epoch. Followers only consume delivered batches.
Override distributed execution error handling to fail all sinks, mark unhealthy,
send `ReplicaFatal`, and stop; never call local `_recover`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
pytest tests/test_tp_control.py tests/test_tp_service.py \
  tests/test_serving_engine.py -q
```

Commit:

```bash
git add auto_infer/serving/tp_control.py auto_infer/serving/tp_service.py \
  tests/test_tp_control.py tests/test_tp_service.py
git commit -m "feat: add change-only SPMD engine control"
```

### Task 4: Supervised production TP server

**Files:**
- Create: `auto_infer/serving/tp_server.py`
- Modify: `auto_infer/entrypoints/cli.py`
- Modify: `auto_infer/deploy/launcher.py`
- Test: `tests/test_tp_server.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `TpServingConfig`
- Produces: `validate_tp_serving(config, model_metadata)`
- Produces: `serve_tp(...)`
- Produces: fail-fast child supervision reusable by the TP server

- [ ] **Step 1: Write failing topology/CLI tests**

Cover `tp_size < 2`, device-count mismatch, duplicate/negative devices,
`graph_mtp`, multi-node input, and CLI forwarding of `--tp-size`, `--devices`,
`--master-port`, and `--tp-watchdog-timeout`.

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/test_tp_server.py tests/test_cli.py -q`

- [ ] **Step 3: Implement validated spawn configuration**

Set visible devices, allocator settings, async HCCL error handling, and AIV
graph expansion before importing torch-npu. Initialize the distributed world
once, construct the configured executor/service per rank, and load the tokenizer
only on rank 0.

- [ ] **Step 4: Implement replica-wide supervision**

Wait for all ready statuses. If a child exits unexpectedly or publishes fatal,
terminate every live sibling, join with a bound, and return a non-zero server
failure. Normal rank-0 shutdown publishes `shutdown` and joins followers.

- [ ] **Step 5: Attach rank 0 to the shared production runtime**

Construct `AsyncEngine.from_service(spmd_service)`, then call the shared
`ApiRuntime`/Uvicorn helpers from Task 2. Do not duplicate HTTP routes or mutate
private globals.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
pytest tests/test_tp_server.py tests/test_cli.py \
  tests/test_text_serving_api.py -q
```

Commit:

```bash
git add auto_infer/serving/tp_server.py auto_infer/entrypoints/cli.py \
  auto_infer/deploy/launcher.py tests/test_tp_server.py tests/test_cli.py
git commit -m "feat: serve supervised tensor-parallel replicas"
```

### Task 5: Graph and accuracy gates

**Files:**
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/harness/capabilities.py`
- Modify: `auto_infer/harness/package.py`
- Create: `scripts/validate_tp_serving.py`
- Test: `tests/test_tp_capabilities.py`

**Interfaces:**
- Produces: `BaseCausalLM.logits_partition(hidden)`
- Produces: explicit model-package TP capability metadata
- Produces: TP1-versus-TPN token-comparison artifact

- [ ] **Step 1: Write failing capability tests**

Generated Qwen packages declare BF16 TP support and quantization reserved;
generated MLA packages declare TP unsupported/EP supported. Graph MTP with
`tp_size > 1` fails before executor allocation.

- [ ] **Step 2: Add the future vocab-parallel seam**

`logits_partition()` returns the resident head projection plus vocabulary range.
`logits()` preserves current replicated behavior. Runners continue calling
`logits()` until distributed greedy has its own benchmark and graph gate.

- [ ] **Step 3: Add the deterministic NPU validation script**

The script sends the same fixed prompt set to TP1 and TPN production endpoints,
records token IDs, text, prefix-cache counters, TTFT, TPOT, throughput, and
process HBM, and exits non-zero on any greedy token mismatch.

- [ ] **Step 4: Run the full host gate and commit**

Run:

```bash
pytest -q
python -m compileall -q auto_infer tests scripts
git diff --check
```

Commit:

```bash
git add auto_infer/models/base.py auto_infer/worker/graph_decode_runner.py \
  auto_infer/harness tests scripts/validate_tp_serving.py
git commit -m "test: gate tensor-parallel serving accuracy"
```

### Task 6: NPU2 `/data2` validation and documentation

**Files:**
- Create: `docs/TP-SERVING-VALIDATION-2026-07-24.md`
- Update: `README.md`

**Interfaces:**
- Consumes: production `serve --tp-size` CLI
- Produces: reproducible commands, raw artifact paths, accuracy verdict, failure
  behavior, and performance measurements

- [ ] **Step 1: Deploy the exact committed source to npu2 `/data2`**

Record commit SHA, Python, torch, torch-npu, CANN, firmware, visible devices,
model path, environment, and command line.

- [ ] **Step 2: Run TP2 small-model BF16 gates**

Run paged and graph modes for prefill, 32-token decode, B1/B4/B16 continuous
batching, prefix-cache hit, abort, and killed-follower failure. Compare every
greedy token with TP1.

- [ ] **Step 3: Run Qwen2.5-72B TP8**

Verify shard-at-read startup memory, production HTTP readiness, concurrent
requests, and sustained decode. If the checkpoint is unavailable, record that
external blocker and retain the executable command rather than fabricating a
result.

- [ ] **Step 4: Profile the control plane**

Capture Chrome-readable traces proving there is no per-step Gloo object
broadcast and report change-only control messages separately from prefill and
decode device execution.

- [ ] **Step 5: Write the validation report, run final tests, and commit**

Run:

```bash
pytest -q
git diff --check
```

Commit:

```bash
git add README.md docs/TP-SERVING-VALIDATION-2026-07-24.md
git commit -m "docs: validate production TP serving"
```
