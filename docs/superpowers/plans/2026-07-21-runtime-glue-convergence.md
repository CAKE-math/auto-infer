# Runtime Glue Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove remaining runtime-glue duplication and dead production paths while preserving the validated BF16 decode, GQA MTP, and low-level P/D contracts.

**Architecture:** Shared staging and graph-task lifecycle helpers own repeated mechanics; attention-family registration owns MTP backend capability; the engine factory alone owns distributed bootstrap. Unwired serving and numerical-probe code leaves the installed runtime, while P/D and MLA MTP remain explicit, accurately bounded interfaces.

**Tech Stack:** Python 3.12, PyTorch/torch-npu, pytest, AST dependency checks, Ascend 910B1.

## Global Constraints

- Preserve BF16 output identity and existing graph event ordering.
- Do not add kernels, allocations, synchronization, or Python branches to captured replay.
- P/D remains a low-level interface; do not add serving integration.
- MLA MTP remains an unsupported registered capability; do not claim execution support.
- Use tests before production edits and keep each task independently green.
- Final GitHub history must return to one root commit on the sole `main` branch.

---

### Task 1: Unify persistent staging primitives

**Files:**
- Modify: `tests/test_decode_input_stager.py`
- Modify: `tests/test_prefill_input_stager.py`
- Modify: `tests/test_mtp_pipeline_stager.py`
- Modify: `tests/test_device_token_batch.py`
- Modify: `tests/test_architecture_convergence.py`
- Modify: `auto_infer/worker/staging.py`
- Modify: `auto_infer/worker/decode_input_stager.py`
- Modify: `auto_infer/worker/prefill_input_stager.py`
- Modify: `auto_infer/worker/mtp_pipeline_stager.py`
- Modify: `auto_infer/worker/model_runner.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`

**Interfaces:**
- Produces: `splice_device_tokens(target, target_rows, request_order, refs)` in `worker.staging`.
- Produces: `upload_dirty_block_table(device, host, current, shadow, non_blocking, *, active_rows=None) -> tuple[int, int]`.

- [ ] **Step 1: Write failing ownership and accounting tests**

Add a continuation-stage regression that changes two block-table rows and asserts
both `copied_block_rows` and `copied_block_elements`. Change the architecture
test to import `splice_device_tokens` from `worker.staging` and assert the old
module no longer owns it. Add an AST assertion that every stager calls
`upload_dirty_block_table` and contains no local `dirty_spans` loop.

- [ ] **Step 2: Run tests and confirm RED**

Run:
`pytest -q tests/test_decode_input_stager.py tests/test_prefill_input_stager.py tests/test_mtp_pipeline_stager.py tests/test_device_token_batch.py tests/test_architecture_convergence.py`

Expected: continuation element accounting and new helper ownership fail.

- [ ] **Step 3: Implement the shared primitives**

Move token splicing to `worker/staging.py`. Implement dirty upload so it computes
the dirty mask, optionally clears rows after `active_rows`, uploads each span,
updates the shadow, and returns `(rows, rows * row_width)`. Replace all five
hand-written loops and add returned counts to each stager's counters.

- [ ] **Step 4: Run the focused tests**

Expected: all focused staging tests pass and no old import remains.

- [ ] **Step 5: Commit**

Commit message: `refactor: unify persistent staging updates`

### Task 2: Remove or relocate non-runtime code

**Files:**
- Delete: `auto_infer/serving/router.py`
- Delete: `tests/test_router.py`
- Delete: `scripts/verify_router.py`
- Move: `auto_infer/serving/sse_client.py` to `benchmarks/sse_client.py`
- Modify: `benchmarks/run_serving_online.py`
- Modify: `scripts/verify_native_async_serving.py`
- Modify: `tests/test_serving_benchmark.py`
- Modify: `auto_infer/layers/moe/fused_moe.py`
- Modify: `scripts/verify_w8a8_moe.py`
- Modify: `auto_infer/distributed/parallel_state.py`
- Modify: `scripts/verify_parallel_mesh_npu.py`
- Modify: `scripts/verify_ep_dispatch.py`
- Modify: `tests/test_verify_ep_dispatch.py`
- Modify: `auto_infer/worker/decode_epilogue.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `tests/test_decode_epilogue.py`
- Modify: `tests/test_architecture_convergence.py`
- Modify: `auto_infer/layers/attention/mla.py`
- Modify: architecture validation documents that overstate P/D integration or zero redundancy

**Interfaces:**
- Keeps: `auto_infer.pd.connector.transfer_hccl` and `copy_blocks` as low-level experimental operations.
- Produces: `worker.decode_epilogue.is_capturable_greedy(requests) -> bool`.

- [ ] **Step 1: Add failing structural tests**

Assert the runtime router and SSE parser modules are absent, W8A8 MoE probe
symbols and `ep_all_reduce` are absent from production, `DecodeEpilogue` is
absent, and MLA contains no self-import or duplicate base imports.

- [ ] **Step 2: Run tests and confirm RED**

Run:
`pytest -q tests/test_architecture_convergence.py tests/test_decode_epilogue.py tests/test_serving_benchmark.py tests/test_verify_ep_dispatch.py tests/test_pd_connector.py`

Expected: structural assertions fail against the current tree.

- [ ] **Step 3: Relocate and delete**

Move SSE parsing to `benchmarks`. Place W8A8 MoE probe math directly in
`scripts/verify_w8a8_moe.py`. Replace verification-only EP reductions with a
script-local helper calling `torch.distributed.all_reduce`. Convert the
epilogue class to a function and remove redundant MLA imports. Delete router
artifacts. Keep P/D operations and label them low-level/experimental in code and
reports.

- [ ] **Step 4: Run focused tests and static searches**

Run the focused tests and:
`rg -n 'serving.router|serving.sse_client|fused_experts_w8a8|build_expert_weights_w8a8|ep_all_reduce|DecodeEpilogue' auto_infer tests scripts benchmarks`

Expected: only negative structural assertions or script-local names remain.

- [ ] **Step 5: Commit**

Commit message: `refactor: remove non-runtime compatibility surfaces`

### Task 3: Put MTP construction behind attention-family capability

**Files:**
- Modify: `tests/test_attention_backend.py`
- Modify: `tests/test_multistep_mtp.py`
- Modify: `tests/test_architecture_convergence.py`
- Modify: `auto_infer/layers/attention/registry.py`
- Modify: `auto_infer/worker/mtp_runner.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`

**Interfaces:**
- Produces: `register_mtp_attention_family(name, builder)` with duplicate rejection.
- Produces: `build_mtp_attention_backend(model, mode, prefix, num_blocks, block_size)`.
- GQA supports `paged` and `graph`; MLA raises an explicit unsupported-capability error.

- [ ] **Step 1: Write failing capability tests**

Test GQA paged/graph builder selection with lightweight fake models, duplicate
MTP family registration rejection, explicit MLA unsupported errors, and AST
absence of concrete GQA backend imports in both MTP runners.

- [ ] **Step 2: Run tests and confirm RED**

Run:
`pytest -q tests/test_attention_backend.py tests/test_multistep_mtp.py tests/test_architecture_convergence.py`

Expected: missing capability API and concrete-import assertions fail.

- [ ] **Step 3: Implement the family capability**

Add a separately guarded MTP builder registry. Its GQA builder creates the
one-layer backend with the supplied prefix and allocates caches. Do not register
an MLA builder. Update eager and graph MTP runners to request the capability.
Translate missing family registration into a precise `NotImplementedError`
naming attention family and mode.

- [ ] **Step 4: Run MTP and attention suites**

Run:
`pytest -q tests/test_attention_backend.py tests/test_multistep_mtp.py tests/test_spec_decode.py tests/test_mtp_pipeline_stager.py`

Expected: all pass.

- [ ] **Step 5: Commit**

Commit message: `refactor: register MTP attention capabilities`

### Task 4: Share graph FIA task lifecycle

**Files:**
- Modify: `tests/test_attention_backend.py`
- Modify: `tests/test_graph_task_pipeline.py`
- Modify: `auto_infer/layers/attention/base.py`
- Modify: `auto_infer/layers/attention/gqa.py`
- Modify: `auto_infer/layers/attention/mla.py`

**Interfaces:**
- Produces: an internal graph-FIA lifecycle helper owning capture state, task entries, begin/end, and update iteration.
- Backend-specific GQA/MLA code continues to own cache views, head counts, dimensions, and FIA invocation.

- [ ] **Step 1: Add failing structural lifecycle tests**

Assert GQA and MLA graph backends inherit/use the shared lifecycle owner and do
not each define duplicate `begin_capture`, `end_capture`, and update-loop state.
Retain ordering tests that record capture/update calls.

- [ ] **Step 2: Run tests and confirm RED**

Run:
`pytest -q tests/test_attention_backend.py tests/test_graph_task_pipeline.py`

Expected: structural ownership assertion fails.

- [ ] **Step 3: Extract lifecycle without changing invocation math**

Move only state-machine mechanics to the shared helper. Keep backend-specific
FIA closures and cache reshaping in their current modules. Reuse existing
`GraphTaskEntry`, `capture_graph_task`, and `update_graph_task`; do not introduce
new tensors or synchronization.

- [ ] **Step 4: Run attention and graph tests**

Expected: all focused tests pass; AST similar-block scan no longer reports the
GQA/MLA lifecycle clusters.

- [ ] **Step 5: Commit**

Commit message: `refactor: share graph FIA task lifecycle`

### Task 5: Establish one distributed bootstrap owner

**Files:**
- Modify: `tests/test_executor_factory.py`
- Modify: `tests/test_engine_core.py`
- Modify: `tests/test_architecture_convergence.py`
- Modify: `auto_infer/engine/factory.py`
- Modify: `auto_infer/engine/engine_core.py`
- Modify: `auto_infer/entrypoints/llm.py` only if injection semantics require an explicit bootstrap call

**Interfaces:**
- `build_executor(config)` remains the production composition root and sole process-group initializer.
- `EngineCore(config, executor)` assumes runtime bootstrap is complete and has no distributed side effect.

- [ ] **Step 1: Write failing ownership tests**

Assert `build_executor` calls `init_distributed` once, direct `EngineCore`
construction does not call it, and only factory owns a production import/call.

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_executor_factory.py tests/test_engine_core.py tests/test_architecture_convergence.py`

Expected: direct EngineCore construction still initializes distributed state.

- [ ] **Step 3: Remove the duplicate owner**

Delete distributed initialization from `EngineCore`. Keep validation in
configuration/mesh construction and idempotence in `parallel_state`.

- [ ] **Step 4: Run engine, distributed, and serving suites**

Run:
`pytest -q tests/test_executor_factory.py tests/test_engine_core.py tests/test_parallel_mesh.py tests/test_sp_ep_mesh.py tests/test_serving_engine.py tests/test_serving_lifecycle.py`

Expected: all pass.

- [ ] **Step 5: Commit**

Commit message: `refactor: centralize distributed bootstrap`

### Task 6: Final structural and device validation

**Files:**
- Modify: `docs/ARCHITECTURE-COMPARISON.md`
- Modify: `docs/FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md`
- Modify: `docs/ARCHITECTURE-CONVERGENCE-VALIDATION-2026-07-21.md`

**Interfaces:**
- Produces a clean single-root tree with retained host/NPU evidence and bounded claims.

- [ ] **Step 1: Run complete host verification**

Run `pytest -q`, compileall, pyflakes, `git diff --check`, tracked whitespace,
internal SCC, forbidden-edge, exact duplicate-body, self-import, and zero-incoming
production-symbol scans.

Expected: all tests pass and every structural scan is empty except explicitly
documented public/tool entry points.

- [ ] **Step 2: Run focused NPU validation on npu2 `/data2`**

Use a free Ascend 910B1. Run BF16 packed-MLA versus independent-segment parity,
then MiMo graph-MTP K1 and K2 against graph-greedy token identity. Reuse retained
benchmark manifests and record exact logs. Abort rather than taking an occupied
device.

- [ ] **Step 3: Run performance smoke**

Require no statistically material regression in the existing Moonlight/MiMo
workload outside retained run-to-run noise. Report measured values; do not alter
thresholds after observing results.

- [ ] **Step 4: Update reports from the final tree**

Regenerate Python file/LOC/complexity counts. State P/D as a low-level interface
and MLA MTP as an unsupported extension capability. Remove all zero-redundancy or
whole-product superiority claims not supported by the final scans.

- [ ] **Step 5: Request final read-only review**

Require zero Critical/Important findings before publishing.

- [ ] **Step 6: Restore single-root history and publish**

Create a parentless commit with the verified final tree, confirm tree identity
and `git rev-list --count HEAD == 1`, then force-push with an exact
`--force-with-lease`. Confirm the CAKE remote exposes only `main`.
