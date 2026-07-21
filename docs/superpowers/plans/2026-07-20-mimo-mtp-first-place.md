# MiMo MTP First-Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional fused two-row MiMo drafter with a two-stage target/compacted-drafter graph pipeline that beats same-run vLLM-Ascend by at least 3% at B4 and B16 while remaining token-identical to auto-infer graph-plain.

**Architecture:** A request-count target graph verifies two rows per request and writes acceptance plus compacted confirmed hidden/token/position/slot buffers. After a pinned acceptance-control copy supplies FIA's Python metadata, a separately prewarmed drafter graph keyed by `(confirmed_token_gear, request_gear)` runs only confirmed rows, updates only confirmed MTP KV, selects one row per request for the shared BF16 head, and emits one packed host result.

**Tech Stack:** Python 3.11, PyTorch/torch-npu ACL graphs, Ascend FIA-v2 graph-task update, pytest, MiMo-7B-Base, Ascend 910B1.

## Global Constraints

- Use one Ascend 910B1 on `npu2`; retain logs under `/data2/auto-infer-decode-performance/logs/mimo-two-stage-20260720`.
- Keep speculative depth K=1, BF16 greedy sampling, 32 output tokens, and the committed four-prompt workload.
- MTP output must be token-identical to auto-infer graph-plain at B4 and B16.
- Auto-infer median throughput must be at least 1.03 times the same-run vLLM-Ascend median at both B4 and B16; CV must be at most 3%.
- Runtime graph lookup must never capture; capture failures for production gears must be zero.
- Rejected target rows must not enter MTP KV. Padding may access only reserved scratch blocks.
- Preserve the current fused MTP graph as a per-gear correctness fallback until the two-stage path passes every gate.

---

### Task 1: Pure confirmed-layout contract

**Files:**
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Test: `tests/test_spec_decode.py`

**Interfaces:**
- Produces: `ConfirmedLayout(source_rows, query_lengths, cumulative_query_lengths, final_rows, active_tokens)`.
- Produces: `_confirmed_layout(accepted: Sequence[int]) -> ConfirmedLayout`.
- Consumes: K=1 acceptance flags containing only `0` or `1`.

- [x] **Step 1: Write failing layout tests**

Add tests for all-reject, all-accept, and mixed layouts. The mixed expectation is:

```python
layout = _confirmed_layout([0, 1, 0, 1])
assert layout.source_rows == (0, 2, 3, 4, 6, 7)
assert layout.query_lengths == (1, 2, 1, 2)
assert layout.cumulative_query_lengths == (1, 3, 4, 6)
assert layout.final_rows == (0, 2, 3, 5)
assert layout.active_tokens == 6
```

Assert values outside `{0, 1}` raise `ValueError`.

- [x] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_spec_decode.py -k confirmed_layout`

Expected: import or assertion failure because the contract does not exist.

- [x] **Step 3: Implement the pure layout helper**

Add a frozen dataclass and a single-pass helper. For request row `r`, always
append source row `2*r`, append `2*r+1` only when accepted, record the cumulative
packed length, and record the final packed row. Reject non-binary flags before
building output.

- [x] **Step 4: Run focused and existing speculative tests**

Run: `pytest -q tests/test_spec_decode.py`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add auto_infer/worker/graph_mtp_runner.py tests/test_spec_decode.py
git commit -m "test: define confirmed MTP row layout"
```

---

### Task 2: Persistent two-stage buffers and host metadata staging

**Files:**
- Create: `auto_infer/worker/mtp_pipeline_stager.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Test: `tests/test_mtp_pipeline_stager.py`

**Interfaces:**
- Produces: `StagedMtpMetadata(block_table, cumulative_query_lengths, kv_lengths, sample_rows, request_count, active_tokens)`.
- Produces: `MtpPipelineStager.stage_drafter(plan, request_order, accepted, token_gear, request_gear)`.
- Owns persistent pinned acceptance-control, block-table, sample-row, and packed-result host buffers.

- [x] **Step 1: Write failing stager tests**

Cover mixed acceptance `[0, 1, 0, 1]`, token gear 8, request gear 4. Assert:

```python
assert staged.cumulative_query_lengths == [1, 3, 4, 6, 8]
assert staged.kv_lengths[:4] == [base0 + 1, base1 + 2, base2 + 1, base3 + 2]
assert staged.sample_rows[:4].tolist() == [0, 2, 3, 5]
assert staged.block_table.shape[0] == 5
assert staged.active_tokens == 6
```

The fifth sequence is a two-token scratch dummy. Add all-accept (no dummy),
all-reject (padding dummy), dirty-row reuse, invalid accepted count, and scratch
isolation tests.

- [x] **Step 2: Run the stager tests and verify RED**

Run: `pytest -q tests/test_mtp_pipeline_stager.py`

Expected: module import failure.

- [x] **Step 3: Implement persistent metadata staging**

Build metadata from `_confirmed_layout`. Real block-table rows come from each
request. If `token_gear > active_tokens`, append exactly one dummy sequence with
length `token_gear - active_tokens`; map it to contiguous scratch blocks. Copy
only dirty block-table spans. Copy sample rows into a persistent device tensor
whose first `request_count` entries select the last confirmed packed row.

- [x] **Step 4: Integrate buffer ownership with graph gears**

Replace ad-hoc two-stage metadata tensors in `graph_mtp_runner.py` with the new
stager. Keep `SpecDecodeInputStager` unchanged for the fused fallback.

- [x] **Step 5: Run focused and full host tests**

Run:

```bash
pytest -q tests/test_mtp_pipeline_stager.py tests/test_spec_decode.py tests/test_decode_input_stager.py
pytest -q
```

Expected: all tests pass.

- [x] **Step 6: Commit**

```bash
git add auto_infer/worker/mtp_pipeline_stager.py auto_infer/worker/graph_mtp_runner.py tests/test_mtp_pipeline_stager.py
git commit -m "feat: stage compacted MTP metadata persistently"
```

---

### Task 3: Startup-prewarmed target and drafter graph families

**Files:**
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Test: `tests/test_spec_decode.py`
- Test: `tests/test_graph_task_pipeline.py`

**Interfaces:**
- Produces: `_TargetGear(request_gear, ...)` keyed by request gear.
- Produces: `_DrafterGear(token_gear, request_gear, ...)` keyed by `(token_gear, request_gear)`.
- Produces: `_select_drafter_gear(active_tokens, request_count, max_gear) -> tuple[int, int] | None`.
- Produces: startup stats `target_capture_attempts`, `drafter_capture_attempts`, `capture_failures`, `online_captures`, `two_stage_steps`, `fused_fallback_steps`.

- [ ] **Step 1: Write failing gear and prewarm tests**

Assert request gear 4 selects target gear 4. Assert `(active=6, requests=4)`
selects drafter `(8, 4)` and `(active=29, requests=16)` selects `(32, 16)`.
Construct a runner stub whose capture methods record calls; assert initialization
prewarms every supported target request gear and reachable drafter pair, while
runtime lookup never invokes capture. Assert one failed pair is isolated and
routes only that pair to fallback.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_spec_decode.py -k "two_stage or drafter_gear or prewarm"`

Expected: failures because the graph families do not exist.

- [ ] **Step 3: Split the current fused body into target and drafter bodies**

Target body:

```python
h_pre = model.forward(target_ctx, prenorm=True)
preds = model.logits(rms_norm(h_pre, norm_weight, eps)).argmax(-1)
accepted = (drafts[:, 0] == preds.view(request_gear, 2)[:, 0]).long()
accepted.mul_(active_mask)
lengths = 1 + accepted
starts = lengths.cumsum(0) - lengths
dest = torch.arange(2 * request_gear, device=accepted.device)
owner = ((dest[:, None] >= starts[None, :]).sum(1) - 1).clamp_min(0)
offset = dest - starts[owner]
source = 2 * owner + offset
active_tokens = lengths.sum()
valid = dest < active_tokens
compact_hidden.copy_(torch.where(valid[:, None], h_pre[source], 0))
compact_tokens.copy_(torch.where(valid, preds.view(-1)[source], 0))
compact_positions.copy_(torch.where(valid, positions[source], scratch_positions))
compact_slots.copy_(torch.where(valid, slots[source], scratch_slots))
```

Drafter body:

```python
mtp_hidden = mtp_layer_hidden(compact_hidden[:token_gear], compact_tokens[:token_gear], ctx)
selected = mtp_hidden.index_select(0, sample_rows[:request_gear])
next_draft = model.logits(rms_norm(selected, final_norm, eps)).argmax(-1)
pack target predictions, accepted flags, and next drafts into result buffer
```

Keep vocabulary heads in BF16 and inside their respective graphs. The drafter
attention uses compacted positions/slots and staged metadata, so rejected rows
are never written to MTP KV.

- [ ] **Step 4: Add explicit event ordering and dual metadata slots**

Give each target and drafter gear its own existing `GraphTaskPipeline`. After
target replay, record `target_done` on the default stream. On a persistent copy
stream, wait for `target_done`, copy the active acceptance prefix to pinned
host memory, and record `control_ready`; wait only for `control_ready` before
deriving drafter metadata and submitting the drafter replay. Maintain two
immutable Python metadata slots per drafter gear. Add host tests proving a
second submission cannot mutate the first submission's lists and that the
drafter submission occurs only after the control event.

- [ ] **Step 5: Prewarm without online capture**

At runner construction, capture target gears `1, 2, 4, 8, 16` up to
`max_gear`. Capture reachable drafter pairs where `request_gear <= token_gear <=
2 * request_gear`, using the canonical flattened-token ladder. Record failures
per pair. Preserve `_fused_body` and `_SpecGear` only as fallback until final
acceptance.

- [ ] **Step 6: Run focused and full host tests**

Run:

```bash
pytest -q tests/test_spec_decode.py tests/test_graph_task_pipeline.py tests/test_mtp_pipeline_stager.py
pytest -q
python -m py_compile auto_infer/worker/graph_mtp_runner.py auto_infer/worker/mtp_pipeline_stager.py
```

Expected: all tests and compilation pass.

- [ ] **Step 7: Commit**

```bash
git add auto_infer/worker/graph_mtp_runner.py tests/test_spec_decode.py tests/test_graph_task_pipeline.py
git commit -m "perf: split target and compacted MTP graphs"
```

---

### Task 4: Packed result handoff and phase evidence

**Files:**
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Modify: `auto_infer/worker/mtp_pipeline_stager.py`
- Modify: `scripts/probe_auto_mtp.py`
- Modify: `scripts/probe_vllm_mtp.py`
- Test: `tests/test_spec_decode.py`
- Test: `tests/test_mtp_pipeline_stager.py`

**Interfaces:**
- Produces one result row `[pred0, pred1, accepted, next_draft]` per request.
- Produces benchmark `phase_counters` and raw B4/B16 samples with digests.

- [ ] **Step 1: Write failing packed-result tests**

Feed packed rows for reject, accept, and request-ending-after-one-token cases.
Assert emitted tokens are `[pred0]` or `[pred0, pred1]`, next drafts retain
request order, and an inactive padded row is never returned.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_spec_decode.py tests/test_mtp_pipeline_stager.py -k packed_result`

Expected: failure because packed result decoding is absent.

- [ ] **Step 3: Implement one final result copy**

Use a persistent pinned pool and copy stream. Wait on the drafter completion
event, copy only `request_count * 4` integers, then decode request dictionaries
on the host. Retain the separate acceptance-control copy required before the
drafter FIA update; do not copy predictions or hidden states at that boundary.

- [ ] **Step 4: Add benchmark evidence**

Make both probes emit batch size, five raw elapsed samples, median throughput,
CV, output digest/token IDs, acceptance, tokens per step, graph capture stats,
and phase counters. Keep input prompts and tokenization byte-for-byte aligned.

- [ ] **Step 5: Run host verification**

Run: `pytest -q && git diff --check`

Expected: all tests pass and diff is clean.

- [ ] **Step 6: Commit**

```bash
git add auto_infer/worker/graph_mtp_runner.py auto_infer/worker/mtp_pipeline_stager.py scripts/probe_auto_mtp.py scripts/probe_vllm_mtp.py tests/test_spec_decode.py tests/test_mtp_pipeline_stager.py
git commit -m "perf: pack two-stage MTP result handoff"
```

---

### Task 5: npu2 correctness, profiling, and performance convergence

**Files:**
- Modify: `docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md`
- Modify only when a measured root cause is proven: files from Tasks 2-4.

**Interfaces:**
- Consumes: `/data1/models/MiMo-7B-Base` and the committed probes.
- Produces: retained B4/B16 auto-infer, graph-plain, and vLLM-Ascend logs.

- [ ] **Step 1: Deploy to an isolated data2 directory**

Rsync the worktree to `/data2/auto-infer-decode-performance` without deleting
retained logs or results. Check `npu-smi info` and select one idle device.

- [ ] **Step 2: Run correctness gates before performance**

Run B4 and B16 MTP plus graph-plain; assert exact token-list equality and 32
tokens per output. Run all-accept/all-reject/mixed diagnostic drafts and late
arrival continuous batching. Assert production capture failures and online
captures are zero.

- [ ] **Step 3: Record per-phase device evidence**

Retain target, acceptance-control, compaction, drafter, result-copy, host, and
total step timings. If the two-stage path misses either throughput gate, change
only the largest measured phase, one hypothesis at a time, with a failing host
test before each production change.

- [ ] **Step 4: Run matched B4 and B16 comparisons**

Run auto-infer and vLLM-Ascend sequentially on the same idle device with one
warmup and five measured iterations. Required inequalities:

```text
auto_B4_median >= 1.03 * vllm_B4_median
auto_B16_median >= 1.03 * vllm_B16_median
auto_B4_CV <= 0.03
auto_B16_CV <= 0.03
```

- [ ] **Step 5: Update retained validation evidence**

Record raw samples, medians, CVs, acceptance, tokens per step, digests, graph
stats, phase timings, exact environment, and log paths in
`docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md`. Do not retain a first-place claim
unless both batch gates pass in the same final code revision.

- [ ] **Step 6: Final verification and review**

Run:

```bash
pytest -q
python -m py_compile auto_infer/worker/graph_mtp_runner.py auto_infer/worker/mtp_pipeline_stager.py
git diff --check
```

Obtain final code review. Fix every Critical and Important issue and rerun the
affected host and NPU gates.

- [ ] **Step 7: Commit final evidence**

```bash
git add docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md
git commit -m "docs: validate first-place MiMo MTP"
```
