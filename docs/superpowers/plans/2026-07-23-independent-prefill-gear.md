# Independent Prefill Gear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate decode batch graph capacity from flattened prefill token graph capacity and regenerate scientifically matched Qwen3 profiling evidence.

**Architecture:** `ExecutionConfig` owns independent `max_gear` and `max_prefill_tokens` limits. The graph executor passes both to `GraphPagedRunner`; decode graph selection uses the former, while prefill enumeration, selection, and scratch capacity use the latter. The profiler rejects an auto-infer capture unless PREFILL actually uses the prewarmed prefill graph.

**Tech Stack:** Python 3.11, dataclasses, PyTorch/torch-npu ACL graphs, pytest, Chrome Trace JSON, Ascend 910B1.

## Global Constraints

- Decode graph capacity remains `max_gear=32` by default.
- Prefill graph capacity defaults to `max_prefill_tokens=256`.
- Prefill graph keys remain flattened-token-only; sequence count is not a key.
- Runtime graph capture remains forbidden.
- Failed or oversized prefill gears retain eager fallback.
- Qwen3 output digests must remain identical across all three frameworks.
- Reports must distinguish profiler-instrumented host ranges from profiler-free headline results.

---

### Task 1: Independent configuration and graph-runner limits

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_executor_factory.py`
- Modify: `tests/test_graph_decode_runner.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/executor_backends.py`
- Modify: `auto_infer/worker/graph_decode_runner.py`

**Interfaces:**
- Produces: `ExecutionConfig.max_prefill_tokens: int = 256`
- Produces: `GraphPagedRunner(..., max_gear: int, max_prefill_tokens: int, ...)`
- Preserves: `_select_gear(B, max_gear)` and `_select_prefill_gear(tokens, max_prefill_tokens)`.

- [ ] Write failing tests with these assertions:

```python
assert ExecutionConfig().max_prefill_tokens == 256
with pytest.raises(ValueError, match="max_prefill_tokens"):
    ExecutionConfig(max_prefill_tokens=0)

config = EngineConfig(
    model=ModelConfig("/models/qwen"),
    execution=ExecutionConfig(mode="graph", max_gear=32,
                              max_prefill_tokens=256))
_, kwargs = executor_arguments(config)
assert kwargs["max_gear"] == 32
assert kwargs["max_prefill_tokens"] == 256
assert _select_gear(33, 32) is None
assert _select_prefill_gear(144, 256) == 144
```

- [ ] Run `pytest -q tests/test_config.py tests/test_executor_factory.py tests/test_graph_decode_runner.py -k 'prefill_tokens or independent_graph_limits'` and confirm it fails because `max_prefill_tokens` is not accepted.
- [ ] Add `max_prefill_tokens: int = 256` to `ExecutionConfig`, validate it is positive, pass it from `_graph`, store it on `GraphPagedRunner`, and replace prefill-only uses of `self.max_gear` with `self.max_prefill_tokens`. Compute scratch capacity as `max(max_gear, max_prefill_tokens)` because both graph families share the scratch block allocation.
- [ ] Run `pytest -q tests/test_config.py tests/test_executor_factory.py tests/test_graph_decode_runner.py tests/test_prefill_input_stager.py` and require zero failures.

### Task 2: CLI and serving propagation

**Files:**
- Modify: `tests/test_serving_cli.py`
- Modify: `auto_infer/entrypoints/cli.py`
- Modify: `auto_infer/serving/api_server.py`

**Interfaces:**
- Produces: `serve(..., max_prefill_tokens: int = 256, ...)`
- Produces: CLI flag `--max-prefill-tokens`.

- [ ] Add this failing parser assertion to `tests/test_serving_cli.py`:

```python
args = build_parser().parse_args([
    "serve", "/model", "--max-gear", "16",
    "--max-prefill-tokens", "192"])
assert args.max_gear == 16
assert args.max_prefill_tokens == 192
```

- [ ] Run `pytest -q tests/test_serving_cli.py -k max_prefill_tokens` and confirm argparse rejects the missing option.
- [ ] Add `serve.add_argument("--max-prefill-tokens", type=int, default=256)`, pass it from `main()` to `serve()`, add the matching `serve` parameter, and pass it to `ExecutionConfig`.
- [ ] Run `pytest -q tests/test_serving_cli.py tests/test_serving_api.py` and require zero failures; if `tests/test_serving_api.py` does not exist, run all files matching `tests/test_serving*.py`.

### Task 3: Profiling path contract

**Files:**
- Modify: `tests/test_qwen3_profile.py`
- Modify: `benchmarks/profile_qwen3.py`
- Modify: `benchmarks/qwen3_profile_common.py`

**Interfaces:**
- Produces: auto-infer capture metadata with runner path counters.
- Produces: validation that PREFILL contains `prefill-graph` and excludes `eager`.

- [ ] Add failing tests to `tests/test_qwen3_profile_report.py`:

```python
assert profile_configuration(_manifest(), "auto-infer")[
    "max_prefill_tokens"] == 256

validate_auto_prefill_path({
    "phases": {"prefill": [{"layer": "prefill-graph"}]}})
with pytest.raises(ValueError, match="prefill graph"):
    validate_auto_prefill_path({
        "phases": {"prefill": [{"layer": "eager"}]}})
```

- [ ] Run `pytest -q tests/test_qwen3_profile_report.py -k 'prefill_tokens or auto_prefill_path'` and confirm failure is caused by the missing workload field/validator.
- [ ] Add `max_prefill_tokens=256` to the workload and auto adapter configuration. Add `validate_auto_prefill_path(call_stack_index)` that requires `prefill-graph` exactly once and rejects `eager`, and invoke it after constructing the call-stack lane for auto-infer.
- [ ] Run `pytest -q tests/test_qwen3_profile_report.py` and require zero failures.

### Task 4: Local verification

**Files:**
- Modify only files required by failures attributable to this change.

**Interfaces:**
- Consumes: all implementation from Tasks 1–3.

- [ ] Run `pytest -q tests/test_config.py tests/test_executor_factory.py tests/test_graph_decode_runner.py tests/test_prefill_input_stager.py tests/test_serving_cli.py tests/test_qwen3_profile_report.py`.
- [ ] Run `pytest -q` and require zero failures.
- [ ] Run `python -m compileall -q auto_infer benchmarks tools tests` and require exit code zero.
- [ ] Run `git diff --check`, then `git diff --stat` and `git diff`; resolve every whitespace error and inspect every changed line.

### Task 5: npu2 recapture and report regeneration

**Files:**
- Replace: `docs/profiling/qwen3/raw/*.trace.json`
- Modify: `docs/profiling/qwen3/summary.json`
- Modify: `docs/profiling/qwen3/manifest.json`
- Modify: `docs/profiling/qwen3/provenance.json`
- Modify: `figures/qwen3-trace-call-stack-comparison.svg`
- Modify: `docs/AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.md`
- Modify: `docs/AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html`
- Modify: generated PDF artifact when tracked by the report workflow.

**Interfaces:**
- Produces: corrected three-framework raw traces and derived reports.

- [ ] Deploy the exact worktree files to npu2 and verify source hashes.
- [ ] Capture auto-infer, vLLM-Ascend, and Omni-NPU on the same physical NPU with the committed workload.
- [ ] Validate output digests, KV capacity, phase counts, auto-infer prefill graph path, trace hashes, and device-event containment.
- [ ] Regenerate analysis, figure, Markdown, HTML, and PDF from the new raw traces.
- [ ] Open the HTML/PDF and visually inspect the corrected prefill/decode call-stack figure.
- [ ] Re-run the full local verification, commit, push main, and verify local/remote commit equality.
