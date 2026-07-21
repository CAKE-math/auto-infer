# Architecture Quality Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the reviewed architecture debt while restoring packed MLA correctness and preserving every supported runtime path.

**Architecture:** Correct behavior first, then move low-level contracts into neutral modules so dependencies point from layers/config toward contracts rather than workers/engine. Preserve the existing model-forward and executor seams; delete only code proven unwired or test-only.

**Tech Stack:** Python 3.11+, PyTorch/torch-npu, pytest, FastAPI, ACL graph tasks.

## Global Constraints

- BF16 remains the only production precision target; quantization interfaces remain available but are not expanded.
- Do not add vLLM compatibility or imports.
- Do not rewrite serving or runner hot paths solely to reduce line count.
- Every behavior change must be observed failing before production code changes.
- Preserve the single-main-branch and final single-root-commit repository policy.

---

### Task 1: Restore packed MLA request isolation

**Files:**
- Modify: `tests/test_attention_backend.py`
- Modify: `auto_infer/layers/attention/mla.py:409-443`

**Interfaces:**
- Consumes: `ForwardContext.cu_seqlens_q: list[int]` cumulative boundaries.
- Produces: `MlaDenseBackend._attn(...) -> Tensor` with independent causal segments.

- [ ] **Step 1: Write the failing packed-MLA test**

Add a test constructing `MlaDenseBackend`, concatenating two random Q/K/V
segments, and asserting the second packed output equals a standalone call:

```python
def test_mla_dense_backend_respects_per_request_segments():
    torch.manual_seed(4)
    t1, t2, heads, qk, vd = 3, 2, 2, 4, 3
    backend = MlaDenseBackend(
        {}, num_heads=heads, qk_nope=2, qk_rope=2,
        v_head_dim=vd, kv_lora_rank=2, q_lora_rank=None,
        rms_eps=1e-6, softmax_scale=qk ** -0.5)
    q = torch.randn(t1 + t2, heads, qk)
    k = torch.randn(t1 + t2, heads, qk)
    v = torch.randn(t1 + t2, heads, vd)
    packed = backend._attn(0, q, k, v, SimpleNamespace(
        cu_seqlens_q=[t1, t1 + t2]))
    isolated = backend._attn(0, q[t1:], k[t1:], v[t1:], SimpleNamespace(
        cu_seqlens_q=[t2]))
    torch.testing.assert_close(packed[t1:], isolated)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest -q tests/test_attention_backend.py::test_mla_dense_backend_respects_per_request_segments`

Expected: failure showing the second segment differs because it attended to the first.

- [ ] **Step 3: Implement segmented MLA dense attention**

Iterate cumulative boundaries, compute causal fp32 attention on each Q/K/V
slice, append each output, and concatenate. Do not branch in paged or graph code.

- [ ] **Step 4: Verify GREEN and surrounding attention tests**

Run: `pytest -q tests/test_attention_backend.py`

Expected: all tests pass.

### Task 2: Correct graph-task and EP topology dependency direction

**Files:**
- Create: `auto_infer/graph_tasks.py`
- Create: `auto_infer/distributed/topology.py`
- Modify: `auto_infer/worker/graph_task_pipeline.py`
- Modify: `auto_infer/layers/attention/gqa.py`
- Modify: `auto_infer/layers/attention/mla.py`
- Modify: `auto_infer/layers/moe/ep_dispatch.py`
- Modify: `auto_infer/distributed/parallel_state.py`
- Modify: `tests/test_architecture_convergence.py`
- Modify: `tests/test_graph_task_pipeline.py`

**Interfaces:**
- Produces: `GraphTaskEntry`, `capture_graph_task`, and `update_graph_task` from `auto_infer.graph_tasks`.
- Produces: `ExpertParallelTopology` from `auto_infer.distributed.topology`.

- [ ] **Step 1: Add failing import-boundary tests**

Parse production AST imports and assert no module under `auto_infer.layers`
imports `auto_infer.worker`, and no module under `auto_infer.distributed` imports
`auto_infer.layers`.

- [ ] **Step 2: Run boundary tests and confirm RED**

Run: `pytest -q tests/test_architecture_convergence.py`

Expected: failure listing attention-to-worker and distributed-to-layer edges.

- [ ] **Step 3: Move neutral contracts**

Move only the entry/capture/update primitives from `worker/graph_task_pipeline.py`
to `auto_infer/graph_tasks.py`; retain `GraphTaskPipeline` in worker. Move the EP
topology dataclass unchanged into `distributed/topology.py` and update imports.

- [ ] **Step 4: Verify graph and EP suites**

Run: `pytest -q tests/test_architecture_convergence.py tests/test_graph_task_pipeline.py tests/test_ep_dispatch.py tests/test_attention_backend.py`

Expected: all tests pass.

### Task 3: Move executor backend ownership out of engine

**Files:**
- Move: `auto_infer/engine/backend_registry.py` -> `auto_infer/executor_backends.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/engine/factory.py`
- Modify: `tests/test_executor_factory.py`
- Modify: `tests/test_architecture_convergence.py`

**Interfaces:**
- Produces: `ExecutorBackend`, `register_executor_backend`,
  `get_executor_backend`, `has_executor_backend` in a neutral top-level module.

- [ ] **Step 1: Add a failing config-boundary test**

Assert config modules do not import `auto_infer.engine` through either top-level
or function-local AST imports.

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest -q tests/test_architecture_convergence.py`

Expected: failure at `config/__init__.py` backend registry import.

- [ ] **Step 3: Move registry and remove duplicate validation wrapper**

Update config, factory, and tests to import the neutral registry. Delete
`validate_execution_config`; tests for invalid combinations call
`executor_arguments`, the production validation path.

- [ ] **Step 4: Verify factory/config behavior**

Run: `pytest -q tests/test_executor_factory.py tests/test_architecture_convergence.py`

Expected: all tests pass.

### Task 4: Unify MTP geometry and construction

**Files:**
- Modify: `tests/test_multistep_mtp.py`
- Modify: `tests/test_spec_decode.py`
- Modify: `auto_infer/spec_decode/geometry.py`
- Modify: `auto_infer/worker/mtp_runner.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`

**Interfaces:**
- Produces: `MtpGeometry.recurrent_from_weights(weights, draft_depth)` accepting
  every positive depth for exactly one recurrent trained layer in eager/paged
  execution; graph execution keeps the retained NPU-verified K2 boundary.
- Produces: `MtpDrafter(head, *, device, block_size)` and
  `MtpDrafter.from_model(model, num_blocks, block_size)`.

- [ ] **Step 1: Change tests to require depth three and normal construction**

Replace the depth-three rejection assertion with:

```python
geometry = MtpGeometry.recurrent_from_weights(weights, 3)
assert geometry.draft_depth == 3
assert geometry.query_width == 4
```

Update the fake drafter helper to call `MtpDrafter(_FakeHead(), device=..., block_size=4)`.
Add an AST/monkeypatch assertion that `from_head` and `__new__` bypasses are absent.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_multistep_mtp.py tests/test_spec_decode.py`

Expected: failures from the old verified-depth cap and missing constructor.

- [ ] **Step 3: Implement one geometry contract**

Remove `_RECURRENT_CONTINUATIONS_PER_TRAINED_LAYER`; retain the requirement for
exactly one trained recurrent layer. Make both runners construct geometry and
enforce graph query-width versus block-size plus the retained NPU token-parity
boundary. Replace the hardcoded MTP prefix with `geometry.layer_prefix(0)`.
Replace `from_head/__new__` with a normal initializer and `from_model` factory.

- [ ] **Step 4: Verify all MTP suites**

Run: `pytest -q tests/test_multistep_mtp.py tests/test_spec_decode.py tests/test_mtp_pipeline_stager.py tests/test_decode_input_stager.py`

Expected: all tests pass.

### Task 5: Harden registries and remove retired production code

**Files:**
- Modify: `tests/test_architecture_convergence.py`
- Modify: `tests/test_executor_factory.py`
- Modify: `tests/test_attention_backend.py`
- Modify: `auto_infer/executor_backends.py`
- Modify: `auto_infer/layers/attention/registry.py`
- Modify: `auto_infer/models/registry.py`
- Delete: `auto_infer/profiling/instrument.py`
- Delete: `auto_infer/engine/errors.py`
- Modify: `auto_infer/layers/moe/fused_moe.py`
- Modify: tests importing `auto_infer.engine.errors`
- Modify: `auto_infer/serving/metrics.py`
- Modify: tests importing `StatLogger` through serving
- Modify: `auto_infer/distributed/parallel_state.py`
- Modify: `tests/test_sp_ep_mesh.py`

**Interfaces:**
- Registry registration raises `ValueError` on duplicate names.
- Public errors are imported only from `auto_infer.errors`.

- [ ] **Step 1: Add failing structural and registry tests**

Assert retired modules/functions are absent. Register the same synthetic name
twice in each registry and assert `ValueError("already registered")`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `pytest -q tests/test_architecture_convergence.py tests/test_executor_factory.py tests/test_attention_backend.py`

Expected: failures for current silent replacement and present retired modules.

- [ ] **Step 3: Remove and harden**

Delete unwired modules/shims, remove `_fused_experts_ep_reference`, update tests
to public imports, remove the serving metrics re-export, replace
`mesh_axis_groups` tests with `ParallelMesh.groups`, and make built-in model
registration exercise the public guarded registry function.

- [ ] **Step 4: Verify affected suites**

Run: `pytest -q tests/test_architecture_convergence.py tests/test_executor_factory.py tests/test_attention_backend.py tests/test_metrics.py tests/test_sp_ep_mesh.py tests/test_ep_dispatch.py`

Expected: all tests pass.

### Task 6: Correct architecture documentation

**Files:**
- Modify: `docs/ARCHITECTURE-COMPARISON.md`
- Modify: `docs/FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md`
- Modify: any validation document containing stale source counts or unqualified leadership claims

**Interfaces:**
- Produces: dated, reproducible source counts and scope-qualified comparisons.

- [ ] **Step 1: Regenerate counts**

Use a small read-only Python command over tracked `*.py` files to record file and
physical-line counts for auto-infer, vLLM-Ascend, and Omni-NPU.

- [ ] **Step 2: Update claims and whitespace**

Replace stale counts and absolute ranking language with evidence: review surface,
composition visibility, vLLM coupling, model adaptation cost, and scope limits.
Replace Markdown trailing-space hard breaks with normal paragraphs or `<br>`.

- [ ] **Step 3: Validate documentation**

Run: `git grep -nE '7,897|82 files|全面吊打|architecturally stronger' -- docs || true`

Expected: no stale or unsupported claims.

### Task 7: Full verification and NPU precision gate

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Produces: a clean, single-root final repository state with host and NPU evidence.

- [ ] **Step 1: Run host verification**

Run: `pytest -q`

Expected: all tests pass.

Run: `python -m compileall -q auto_infer scripts tests`

Expected: exit zero.

Run: dependency SCC/import-boundary audit and full tracked-file whitespace scan.

Expected: zero SCCs, zero forbidden dependency edges, zero whitespace errors.

- [ ] **Step 2: Run focused NPU packed-MLA parity**

On NPU2 `/data2`, compare two packed MLA segments against independent dense
calls. Require max absolute error within the same BF16 tolerance used by the
existing dense parity gate and token identity at sampled rows.

- [ ] **Step 3: Review the final diff**

Confirm every deletion has zero remaining importer, no experimental monkeypatch
is present, and implementation changes match the approved design.

- [ ] **Step 4: Restore the single-root history and push**

Squash the implementation and design into one root commit, verify its tree is
identical before updating `main`, then push the single branch with
`--force-with-lease`.
