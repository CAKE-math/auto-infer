# Final Architecture Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge auto-infer to one clear implementation per supported behavior while preserving host and Ascend correctness, stability, and performance.

**Architecture:** Centralize request lifecycle mechanics in `EngineCore`, replace executor and attention switches with registrations, move MTP layout and staging primitives below runners, then delete the fused MTP and unused compatibility surfaces after device coverage passes. Every structural change starts with a focused failing test and ends with full regression checks.

**Tech Stack:** Python 3.11, PyTorch, torch_npu, pytest, Ascend CANN.

## Global Constraints

- Preserve the public `EngineConfig`, `LLM`, `EngineService`, and executor behavior.
- Do not add dependencies.
- Keep graph tensor addresses and asynchronous ownership stable.
- Do not remove the fused MTP fallback before NPU2 gear coverage succeeds.
- Retain all NPU2 logs below `/data2/auto-infer-decode-performance/logs/final-architecture-20260720`.

---

### Task 1: Execution backend registry

**Files:**
- Modify: `auto_infer/engine/factory.py`
- Modify: `auto_infer/config/__init__.py`
- Test: `tests/test_executor_factory.py`

**Interfaces:**
- Produces: `register_executor_backend(mode, specification)` and registry-driven `executor_arguments` / `build_executor`.

- [ ] Write tests that register a temporary backend and verify validation, argument derivation, and construction without editing factory branches.
- [ ] Run the focused test and verify it fails because registration is absent.
- [ ] Implement immutable backend specifications and register the four built-in modes.
- [ ] Run `tests/test_executor_factory.py` and `tests/test_config.py`.

### Task 2: Attention family registry

**Files:**
- Modify: `auto_infer/layers/attention/registry.py`
- Test: `tests/test_attention_backend.py`

**Interfaces:**
- Produces: `register_attention_family(name, builder)` and `build_attention_backend(model, mode, num_blocks=0, block_size=0)`.

- [ ] Write a test that registers and constructs a synthetic attention family.
- [ ] Verify the test fails on the current dispatcher.
- [ ] Extract GQA/MLA builders and replace the family `if/elif` with lookup.
- [ ] Run attention and model-forward host tests.

### Task 3: Shared MTP layout and staging primitives

**Files:**
- Create: `auto_infer/spec_decode/layout.py`
- Create: `auto_infer/worker/staging.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Modify: `auto_infer/worker/mtp_pipeline_stager.py`
- Modify: `auto_infer/worker/decode_input_stager.py`
- Modify: `auto_infer/worker/prefill_input_stager.py`
- Test: `tests/test_spec_decode.py`
- Test: `tests/test_mtp_pipeline_stager.py`
- Test: `tests/test_input_buffers.py`

**Interfaces:**
- Produces: `ConfirmedLayout`, `confirmed_layout(accepted)`, `PinnedHostBuffers`, and `dirty_spans(dirty)`.

- [ ] Change tests to import layout from `spec_decode.layout` and assert the stager no longer imports the runner.
- [ ] Verify those tests fail.
- [ ] Move the value object and function without changing results.
- [ ] Consolidate pinned host allocation and dirty-span iteration behind shared helpers.
- [ ] Run all stager, graph runner, and spec-decode tests.

### Task 4: Engine request lifecycle convergence

**Files:**
- Modify: `auto_infer/engine/engine_core.py`
- Test: `tests/test_engine_core.py`
- Test: `tests/test_spec_decode.py`

**Interfaces:**
- Produces internal `_schedule_with_preemption`, `_finish_requests`, and shared token/metrics commit helpers.

- [ ] Add focused tests for identical cleanup, preemption, TTFT, and missing-token behavior across sync and MTP paths.
- [ ] Verify at least one test exposes the duplicated/inconsistent behavior.
- [ ] Extract scheduling and completion primitives while leaving async queue ownership explicit.
- [ ] Make sync and MTP methods thin policy adapters over the shared lifecycle.
- [ ] Run all engine, scheduler, request, metrics, and async tests.

### Task 5: Remove zero-production-reference code

**Files:**
- Modify: `auto_infer/models/deepseek_v2.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/decode_input_stager.py`
- Delete: `auto_infer/models/deepseek_mtp.py`
- Delete: `auto_infer/pd/mooncake_transport.py`
- Modify corresponding tests/tools/scripts/imports.

**Interfaces:**
- Preserves the unified model forward, supported HCCL P/D path, and production staging APIs.

- [ ] Add architecture tests asserting removed private/legacy symbols are absent and production modules do not import test tracing hooks.
- [ ] Verify the tests fail.
- [ ] Delete legacy forward, unused marshaller, unused rejection helper, demo-only MTP wrapper, and half-wired Mooncake transport; update or remove their non-production consumers.
- [ ] Mark required torch_npu side-effect imports explicitly and remove genuinely unused bindings.
- [ ] Run the full host suite and pyflakes.

### Task 6: Retire fused MTP graph fallback

**Files:**
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Modify: `tests/test_two_stage_mtp.py` or the existing MTP runner tests selected during implementation.
- Modify: `scripts` verification entry points as needed.

**Interfaces:**
- The two-stage target/drafter pipeline becomes the sole graph decode implementation.

- [ ] Add a host structural test proving no fused fallback symbols or counters remain.
- [ ] On NPU2, run startup capture for every reachable `(request_gear, token_gear)` and retain the log.
- [ ] Run Moonlight B1/B4/B16 token parity, 200-request continuous batching, and repeated capture/replay stability.
- [ ] Only after those gates pass, remove `_SpecGear`, fused capture/replay methods, fallback routing, and compatibility comments.
- [ ] Repeat the full NPU2 correctness and stability matrix.

### Task 7: Final quality and competitor comparison

**Files:**
- Modify: `docs/ARCHITECTURE-COMPARISON.md`
- Create: `docs/FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md`

**Interfaces:**
- Produces retained, reproducible evidence for the final claims.

- [ ] Run full host tests, `compileall`, and pyflakes with zero unexplained findings.
- [ ] Run sequential Moonlight comparisons against vllm-ascend and omni-npu on one idle NPU using the shared manifest.
- [ ] Record source revisions, commands, raw samples, correctness gates, stability results, and log paths.
- [ ] Re-run the architecture audit for duplicate AST bodies, dependency cycles, and zero-reference definitions.
- [ ] Accept “全面领先” only for dimensions supported by the retained evidence; state scope explicitly.
