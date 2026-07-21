# vLLM-Ascend First-Place Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make auto-infer first on TTFT, equal-capacity memory, throughput, TPOT, load time, and normalized stability against vLLM-Ascend.

**Architecture:** Keep one `GraphPagedRunner` and one KV cache, add shape-keyed prefill gears beside decode gears, make model-owned tied/logits storage explicit, and make benchmarks reject unequal KV capacity. Every behavior change is introduced by a failing host test and then verified on NPU2.

**Tech Stack:** Python 3.11, PyTorch, torch-npu ACL Graph/FIA-v2, pytest, Ascend 910B1.

## Global Constraints

- Preserve public `BatchPlan`, `ExecutionResult`, request, and serving APIs.
- Preserve greedy output tokens and all non-greedy fallback behavior.
- Compare peak allocated memory only at identical usable KV token capacity.
- Use at least 20 measured samples for final stability rankings.
- Do not retain a default optimization that fails correctness or regresses its target metric.

---

### Task 1: Correct tied weights and bounded logits precision

**Files:**
- Modify: `auto_infer/models/qwen2.py`
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `tests/test_weight_loader.py`
- Modify: `tests/test_decode_epilogue.py`

**Interfaces:**
- `Qwen2Model.from_pretrained(...)` guarantees pointer aliasing when `cfg.tie_word_embeddings` is true.
- `BaseCausalLM.logits(hidden, out=None, precision=None)` uses resident weights and never caches a full FP32 head.

- [ ] Add a checkpoint containing distinct embedding and head tensors and assert that a tied configuration aliases the embedding pointer while an untied configuration preserves the head.
- [ ] Run `pytest -q tests/test_weight_loader.py` and witness the tied-pointer test fail.
- [ ] Implement alias installation after TP sharding and remove the persistent `_lm_head_fp32` cache.
- [ ] Add BF16/mixed-output tests covering fixed output buffers and FP32-reference argmax parity.
- [ ] Run `pytest -q tests/test_weight_loader.py tests/test_decode_epilogue.py` and the full host suite.
- [ ] On NPU2, run 256-step B1/B16 token parity before accepting the fast policy.
- [ ] Commit with `perf: eliminate duplicate language-model heads`.

### Task 2: Make graph memory proportional to live requirements

**Files:**
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `tests/test_graph_decode_runner.py`
- Modify: `benchmarks/run_auto_infer.py`

**Interfaces:**
- `_scratch_blocks_for_gears(max_gear: int) -> int` returns the largest capturable gear, which is the minimum safe capacity for lazy capture.
- Benchmark reports `usable_kv_tokens`, `physical_kv_blocks`, and `scratch_kv_blocks`.

- [ ] Add failing tests for scratch sizing and BF16 static logits buffers.
- [ ] Run `pytest -q tests/test_graph_decode_runner.py tests/test_benchmark_manifest.py` and witness failures.
- [ ] Allocate only required scratch blocks and use the model logits dtype for graph buffers.
- [ ] Add explicit capacity fields to auto-infer reports.
- [ ] Run focused and full host tests.
- [ ] On NPU2, compare peak allocation at exactly 14,464 usable tokens with vLLM-Ascend.
- [ ] Commit with `perf: reduce graph runner memory footprint`.

### Task 3: Capture prefill and mixed batches

**Files:**
- Create: `auto_infer/worker/prefill_input_stager.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/graph_task_pipeline.py`
- Modify: `tests/test_graph_decode_runner.py`
- Create: `tests/test_prefill_input_stager.py`
- Create: `scripts/verify_prefill_graph.py`

**Interfaces:**
- `PrefillGearKey(query_tokens: int, sequences: int)` identifies the exact fixed graph shape required by FIA-v2.
- `PrefillInputStager.stage(plan, gear) -> StagedPrefillInput` fills fixed buffers and returns sample rows/order.
- `GraphPagedRunner._prefill_graph_submit(plan, gear, prev_sampled)` returns the same handle shape as eager/decode paths.

- [ ] Add failing pure-host tests for gear selection, cumulative query lengths, padding isolation, sample-row selection, stable addresses, and eager fallback.
- [ ] Run `pytest -q tests/test_prefill_input_stager.py tests/test_graph_decode_runner.py` and witness failures.
- [ ] Implement the fixed-address prefill stager without NPU-specific logic.
- [ ] Implement lazy prefill gear warm-up/capture using the shared backend, cache, task pipeline, logits policy, and greedy epilogue.
- [ ] Run focused and full host tests.
- [ ] On NPU2, verify B1/B16 first-token parity, alternating prefill/decode gears, chunked-prefill fallback, preemption, and 200 repeated requests.
- [ ] Profile TTFT and retain the graph path only if median TTFT improves.
- [ ] Commit with `perf: capture paged prefill execution`.

### Task 4: Enforce fair capacity and normalized stability

**Files:**
- Modify: `benchmarks/common.py`
- Modify: `benchmarks/run_auto_infer.py`
- Modify: `benchmarks/run_vllm.py`
- Modify: `benchmarks/comparison_manifest.json`
- Modify: `tests/test_benchmark_manifest.py`
- Modify: `docs/ARCHITECTURE-COMPARISON.md`
- Create: `docs/VLLM-FIRST-VALIDATION-2026-07-20.md`

**Interfaces:**
- `summarize(samples)` additionally returns `count` and `coefficient_of_variation`.
- Every framework result reports `usable_kv_tokens`; comparison rejects mismatches.

- [ ] Add failing schema/statistics tests, including zero-mean CV and unequal-capacity rejection.
- [ ] Run `pytest -q tests/test_benchmark_manifest.py` and witness failures.
- [ ] Implement statistics and capacity validation without removing existing result fields.
- [ ] Increase final measured runs to 20 and record derived elapsed-time samples.
- [ ] Run focused/full host tests and compile checks.
- [ ] Run complete NPU2 correctness, stability, and sequential three-framework comparison.
- [ ] Record commands, raw samples, output digests, capacity, occupancy, and rankings in the validation document.
- [ ] Commit with `docs: validate first-place convergence`.
