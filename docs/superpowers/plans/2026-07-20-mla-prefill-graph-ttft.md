# MLA Prefill Graph TTFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prewarm a vLLM-compatible one-dimensional prefill graph family so Moonlight cold and warmed TTFT beat 46.58 ms without multiplying graphs by sequence count.

**Architecture:** `GraphPagedRunner` owns seven flattened-token prefill gears at `max_gear=32`. Each gear has maximum-capacity persistent buffers; runtime passes the exact active block-table view and dynamic TND metadata through graph-task update, while padding is an independent scratch-only dummy sequence. Capture occurs during engine initialization and runtime only replays or falls back eagerly.

**Tech Stack:** Python 3.11, PyTorch/torch-npu ACL graphs, Ascend FIA-v2, pytest, Ascend 910B1.

## Global Constraints

- Capture sizes follow vLLM/Omni: `[1, 2, 4]`, multiples of 8 below 256, then multiples of 16, truncated by `max_gear`.
- `max_gear=32` must attempt exactly `[1, 2, 4, 8, 16, 24, 32]`.
- Sequence count must not enter the graph key.
- Online requests must never synchronously capture a prefill graph.
- Padding must not write live KV or alter real sampled rows.
- A failed gear falls back eagerly without aborting initialization.
- Async scheduler, MLA math, MoE routing, dtype, and decode graph behavior remain unchanged.

---

### Task 1: One-dimensional vLLM-compatible prefill gears

**Files:**
- Modify: `tests/test_graph_decode_runner.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`

**Interfaces:**
- Produces: `_prefill_capture_sizes(max_gear: int) -> list[int]`
- Produces: `_select_prefill_gear(query_tokens: int, max_gear: int) -> int | None`
- Removes: sequence count from the internal prefill graph key.

- [x] **Step 1: Write failing gear-policy tests**

```python
def test_prefill_capture_sizes_match_vllm_policy():
    assert _prefill_capture_sizes(32) == [1, 2, 4, 8, 16, 24, 32]
    assert len(_prefill_capture_sizes(256)) == 35


@pytest.mark.parametrize("tokens,expected", [
    (1, 1), (3, 4), (10, 16), (17, 24), (25, 32), (33, None),
])
def test_select_prefill_gear_uses_flattened_token_count(tokens, expected):
    assert _select_prefill_gear(tokens, max_gear=32) == expected
```

- [x] **Step 2: Run the tests and verify RED**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_graph_decode_runner.py -k "prefill_capture_sizes or select_prefill_gear"`

Expected: FAIL because `_prefill_capture_sizes` does not exist and the old selector returns `PrefillGearKey` keyed by sequence count.

- [x] **Step 3: Implement the minimal one-dimensional policy**

```python
def _prefill_capture_sizes(max_gear):
    sizes = [1, 2, 4]
    sizes += list(range(8, min(max_gear + 1, 256), 8))
    if max_gear >= 256:
        sizes += list(range(256, max_gear + 1, 16))
    return sorted({size for size in sizes if size <= max_gear})


def _select_prefill_gear(query_tokens, max_gear):
    return next((size for size in _prefill_capture_sizes(max_gear)
                 if size >= query_tokens), None)
```

Change `prefill_gears` and `failed_prefill_gears` to use integer token gears.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_graph_decode_runner.py -k "prefill_capture_sizes or select_prefill_gear"`

Expected: all selected tests pass.

---

### Task 2: Persistent padded prefill staging

**Files:**
- Modify: `tests/test_prefill_input_stager.py`
- Modify: `auto_infer/worker/prefill_input_stager.py`

**Interfaces:**
- Extends: `StagedPrefillInput` with `real_query_tokens: int` and `sequence_count: int`.
- Changes: `StagedPrefillInput.block_table` is the exact persistent active-row view (`[:B]` or `[:B + 1]` with a padding dummy).
- Preserves: `data_ptrs()`, dirty-row accounting, `sample_order`, and `splice_order`.

- [x] **Step 1: Write failing padding and dynamic-sequence tests**

```python
def test_prefill_stager_pads_to_token_gear_without_sampling_padding():
    stager = _stager(query_gear=8, sequence_gear=8)
    plan = _Plan(
        [_item("a", 3), _item("b", 2)],
        {"a": _request([10, 11, 12], 0, 3),
         "b": _request([20, 21], 0, 2)},
        {"a": (5,), "b": (7,)})
    staged = stager.stage(plan)
    assert staged.real_query_tokens == 5
    assert staged.query_tokens == 8
    assert staged.sequence_count == 2
    assert tuple(staged.block_table.shape) == (3, 4)
    assert staged.sample_rows[:2].tolist() == [2, 4]
    assert staged.cumulative_query_lengths == [3, 5, 8]
    assert staged.kv_lengths == [3, 2, 3]
    assert staged.sample_order == ["a", "b"]
    assert all(slot >= 100 * 4 for slot in staged.slots[5:].tolist())
```

Add a second test that stages one and two sequences through the same 8-token
stager, checks `block_table.shape[0]` is respectively 1 and 2, and checks both
views retain the same base `data_ptr`. Add a third test where padding crosses a
block-table boundary and assert the extra entry references scratch.

- [x] **Step 2: Run the stager tests and verify RED**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_prefill_input_stager.py`

Expected: FAIL because staging currently requires exact query and sequence shapes.

- [x] **Step 3: Implement padded persistent staging**

Update `stage` to reject only `real_query_tokens > query_gear` or
`B > sequence_capacity`. Record every real sample row before appending
`query_gear - real_query_tokens` rows to one independent dummy sequence. Give
the dummy zero-based positions plus scratch-only slots and block-table entries,
append its cumulative query/KV metadata, copy only dirty rows, and return
`self.block_table[:B]` without padding or `self.block_table[:B + 1]` with it.

Keep `sample_rows` at fixed `query_gear` capacity: fill its first `B` entries
with real sample rows and clear the remainder to zero so the captured fixed-size
lm-head/argmax never reads an invalid index.

- [x] **Step 4: Run the stager tests and verify GREEN**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_prefill_input_stager.py`

Expected: all stager tests pass.

---

### Task 3: MLA capability and startup-only graph prewarm

**Files:**
- Modify: `tests/test_graph_decode_runner.py`
- Modify: `auto_infer/layers/attention/mla.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`

**Interfaces:**
- Adds: `GraphMlaBackend.supports_prefill_graph = True`.
- Adds: `GraphPagedRunner._prewarm_prefill_gears() -> None`.
- Changes: `_PrefillGear` is keyed only by token gear and has token-gear-sized block-table/sample/logit buffers.
- Changes: `_get_prefill_gear(gear: int)` is lookup-only at runtime.

- [x] **Step 1: Write failing capability and prewarm-policy tests**

```python
def test_graph_mla_backend_advertises_prefill_graph_support():
    assert GraphMlaBackend.supports_prefill_graph is True


def test_prefill_prewarm_attempts_each_token_gear_once():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner.max_gear = 32
    runner.prefill_gears = {}
    runner.failed_prefill_gears = set()
    runner.stats = {"prefill_graph_capture_attempts": 0,
                    "prefill_graph_capture_failures": 0}
    attempted = []
    runner._capture_prefill = lambda gear: attempted.append(gear) or object()
    runner._prewarm_prefill_gears()
    assert attempted == [1, 2, 4, 8, 16, 24, 32]
    assert sorted(runner.prefill_gears) == attempted


def test_runtime_prefill_lookup_never_captures():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner.prefill_gears = {16: object()}
    runner.failed_prefill_gears = {24}
    runner._capture_prefill = lambda gear: pytest.fail("online capture")
    assert runner._get_prefill_gear(16) is runner.prefill_gears[16]
    assert runner._get_prefill_gear(24) is None
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_graph_decode_runner.py -k "advertises_prefill or prewarm or runtime_prefill_lookup"`

Expected: FAIL because MLA lacks the capability and runtime currently captures lazily.

- [x] **Step 3: Implement startup capture and fixed-size output selection**

Capture every token gear with `query_gear` one-token dummy sequences, scratch
block-table rows, and fixed `sample_rows` length `query_gear`. Capture model,
lm-head, and argmax for all selected rows. During replay, copy runtime sample
rows into the persistent index tensor, pass `staged.block_table` (the exact
`[:B]` view) to graph-task update, and expose only `gear.sampled[:B]`.

Call `_prewarm_prefill_gears` from runner initialization only when the backend
advertises support and `force_eager` is false. Catch capture errors per gear,
record failures, and keep runtime `_get_prefill_gear` lookup-only.

- [x] **Step 4: Run graph-runner and stager tests and verify GREEN**

Run: `/opt/anaconda3/bin/python -m pytest -q tests/test_graph_decode_runner.py tests/test_prefill_input_stager.py`

Expected: all tests pass.

---

### Task 4: Full verification and Moonlight NPU acceptance

**Files:**
- Modify: `docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md`

**Interfaces:**
- Consumes: `benchmarks/moonlight_manifest.json`.
- Produces: cold/warmed TTFT, load time, B4 throughput/CV, path counters, output digest, and continuous-batching evidence.

- [x] **Step 1: Run the complete host suite**

Run: `/opt/anaconda3/bin/python -m pytest -q`

Expected: all tests pass with zero failures.

- [x] **Step 2: Deploy and run Moonlight correctness on npu2**

Rsync the worktree with `.git`, caches, logs, and results excluded. Run graph
and forced-eager B1/B4 32-token generations on one free Ascend 910B1 and assert
the token lists and digest match. Run the existing mixed continuous-batching
probe and assert every request completes with the expected output length.

- [x] **Step 3: Run cold and warmed performance acceptance**

Measure engine construction separately. Immediately after construction run one
B1 one-token request, then one warmup and five measured B1/B4 iterations using
the committed manifest. Assert:

```text
cold post-init TTFT < 46.58 ms
warmed median TTFT < 46.58 ms
warmed TTFT CV < 5%
B4 median throughput >= 219.15 tok/s
load time < 69.07 s
prefill_graph_steps > 0
prefill graph capture attempts == 7
online prefill captures == 0
```

- [x] **Step 4: Record evidence and run final checks**

Update the Moonlight report with exact samples, graph counts, digest, startup,
and memory deltas. Run:

```bash
/opt/anaconda3/bin/python -m pytest -q
/opt/anaconda3/bin/python -m py_compile auto_infer/layers/attention/mla.py auto_infer/worker/graph_decode_runner.py auto_infer/worker/prefill_input_stager.py
git diff --check
```

Expected: all tests pass, compilation exits zero, and diff check is clean.

- [x] **Step 5: Commit the verified implementation**

```bash
git add auto_infer/layers/attention/mla.py auto_infer/worker/graph_decode_runner.py auto_infer/worker/prefill_input_stager.py tests/test_graph_decode_runner.py tests/test_prefill_input_stager.py docs/MTP-MOONLIGHT-VALIDATION-2026-07-20.md
git commit -m "perf: prewarm MLA prefill token gears"
```
