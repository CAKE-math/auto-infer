# Qwen3 Architecture and Performance HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a management-readable, engineering-auditable HTML report with matched Qwen3 results and three directly openable Chrome Trace profiler files.

**Architecture:** Keep profiling outside the inference runtime. A single benchmark launcher adapts auto-infer and vLLM-compatible engines to one capture contract; a pure-Python trace analyzer normalizes Chrome Trace events; a deterministic report builder consumes existing benchmark JSON plus normalized profile JSON and emits one offline HTML file.

**Tech Stack:** Python 3.11+, `torch_npu.profiler`, Chrome Trace Event JSON, standard-library JSON/hash/statistics/HTML generation, pytest, native HTML/CSS/SVG, npu2 Ascend 910B1.

## Global Constraints

- Headline Qwen3 metrics come only from the existing unprofiled matched benchmark: Qwen3-0.6B, BF16, greedy/ignore-EOS, B1 latency, B16 throughput, 128 output tokens, one warm-up, twenty measurements, and 14,464 usable KV tokens.
- Profiled timings must be labeled profiler-instrumented and must not replace TTFT, TPOT, throughput, load, memory, or CV headline values.
- auto-infer and vllm-ascend may claim validated digest identity; omni-npu may claim output-length parity and performance comparability only.
- Raw profiler output must remain uncompressed Chrome Trace JSON that opens in Perfetto or `chrome://tracing`.
- Capture each framework sequentially on the same idle Ascend 910B1 and preserve exact environment/version metadata.
- Unknown trace events remain `unclassified`; the analyzer must never silently assign them to a favorable phase.
- No profiler hooks or report-specific branches may enter `auto_infer/` production runtime.
- The final report is Chinese, responsive, printable, offline, and has no CDN or JavaScript dependency.

---

### Task 1: Define the trace artifact contract

**Files:**
- Create: `benchmarks/qwen3_profile_common.py`
- Create: `tests/test_qwen3_profile_report.py`

**Interfaces:**
- Produces: `validate_chrome_trace(path: Path) -> dict`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `write_profile_metadata(path: Path, metadata: dict) -> None`
- Consumes later: profiler launchers and trace analyzer use the same artifact validator.

- [ ] **Step 1: Write failing artifact-contract tests**

```python
def test_validate_chrome_trace_accepts_trace_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [
        {"name": "GraphReplay", "ph": "X", "ts": 10, "dur": 4,
         "pid": 1, "tid": 2}
    ]}))
    result = validate_chrome_trace(path)
    assert result == {"event_count": 1, "size_bytes": path.stat().st_size}


def test_validate_chrome_trace_rejects_missing_event_array(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="traceEvents"):
        validate_chrome_trace(path)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_qwen3_profile_report.py`

Expected: import failure for `benchmarks.qwen3_profile_common`.

- [ ] **Step 3: Implement the minimal artifact contract**

```python
def validate_chrome_trace(path: Path) -> dict:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    return {"event_count": len(events), "size_bytes": path.stat().st_size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

`write_profile_metadata` must create parent directories, serialize sorted JSON,
and append one newline. It must reject missing `framework`, `trace`, `workload`,
`environment`, `output_digest`, and `output_length` keys.

- [ ] **Step 4: Run contract tests**

Run: `pytest -q tests/test_qwen3_profile_report.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/qwen3_profile_common.py tests/test_qwen3_profile_report.py
git commit -m "test: define Qwen3 profile artifact contract"
```

### Task 2: Add one matched Qwen3 profiler launcher

**Files:**
- Create: `benchmarks/profile_qwen3.py`
- Modify: `benchmarks/qwen3_profile_common.py`
- Modify: `tests/test_qwen3_profile_report.py`

**Interfaces:**
- Consumes: `validate_chrome_trace`, `sha256_file`, `write_profile_metadata`.
- Produces: CLI `python benchmarks/profile_qwen3.py MANIFEST FRAMEWORK OUTPUT_DIR`.
- Produces: `OUTPUT_DIR/FRAMEWORK.trace.json` and `OUTPUT_DIR/FRAMEWORK.metadata.json`.

- [ ] **Step 1: Write failing CLI/config tests**

```python
def test_profile_configuration_is_bounded_and_matched():
    config = profile_configuration(_manifest(), "auto-infer")
    assert config["batch_size"] == 16
    assert config["output_tokens"] == 16
    assert config["warmup_runs"] == 1
    assert config["usable_kv_tokens"] == 14464


@pytest.mark.parametrize("framework", ["auto-infer", "omni-npu", "vllm-ascend"])
def test_profile_configuration_accepts_supported_frameworks(framework):
    assert profile_configuration(_manifest(), framework)["framework"] == framework
```

The profile capture uses 16 output tokens rather than the 128-token headline
request so trace files remain bounded, while retaining B16 and the exact KV,
dtype, prompt, sampling, and model configuration.

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_qwen3_profile_report.py -k profile_configuration`

Expected: missing `profile_configuration`.

- [ ] **Step 3: Implement adapters without runtime changes**

`profile_qwen3.py` must:

1. load and validate `comparison_manifest.json`;
2. select auto-infer or vLLM-compatible construction after parsing the CLI;
3. warm up one B16 eight-token request outside the profiler;
4. construct `torch_npu.profiler.profile` with CPU and NPU activities,
   `record_shapes=False`, `profile_memory=False`, `with_stack=False`;
5. wrap exactly one B16 16-token generation in `record_function` ranges named
   `qwen3/profiled_request`, `qwen3/prefill_and_decode`, and framework name;
6. synchronize before entering and before leaving the profiler;
7. call `export_chrome_trace` after the context closes;
8. emit output digest/length, runtime capacity, environment versions, source
   revision, trace hash/size/event count, and capture timestamps.

The vLLM-compatible adapter must preserve:

```python
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
LLM(..., enforce_eager=False, kv_cache_memory_bytes=manifest["kv_cache_memory_bytes"])
```

The omni invocation remains controlled by its existing plugin environment and
the compatibility `sys.argv[2]` model slot. The launcher must not set omni
plugins implicitly; the command records them from the environment.

- [ ] **Step 4: Run host tests and CLI help**

Run: `pytest -q tests/test_qwen3_profile_report.py`

Run: `python benchmarks/profile_qwen3.py --help`

Expected: tests pass and help lists manifest/framework/output directory.

- [ ] **Step 5: Run a one-event NPU profiler probe**

On npu2, run a tiny tensor operation through the same profiler constructor and
export a temporary Chrome trace. Validate it with `validate_chrome_trace`.

Expected: trace JSON contains both host and NPU events and opens with the parser.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/profile_qwen3.py benchmarks/qwen3_profile_common.py tests/test_qwen3_profile_report.py
git commit -m "feat: add matched Qwen3 profiler launcher"
```

### Task 3: Normalize Chrome traces into comparable phase summaries

**Files:**
- Create: `tools/analyze_qwen3_profiles.py`
- Modify: `tests/test_qwen3_profile_report.py`

**Interfaces:**
- Produces: `classify_event(name: str) -> str`.
- Produces: `summarize_trace(path: Path) -> dict`.
- Produces: CLI accepting three metadata JSON files plus three benchmark JSON
  files and writing `docs/profiling/qwen3/manifest.json` and `summary.json`.

- [ ] **Step 1: Write failing classification and aggregation tests**

```python
@pytest.mark.parametrize(("name", "phase"), [
    ("GraphReplay", "graph_replay"),
    ("npu_fused_infer_attention_score_v2", "attention_kv"),
    ("npu_scatter_pa_kv_cache", "attention_kv"),
    ("npu_grouped_matmul", "projection_mlp_norm"),
    ("aten::argmax", "lm_head_sampling"),
    ("unknown_vendor_op", "unclassified"),
])
def test_classify_event(name, phase):
    assert classify_event(name) == phase


def test_trace_summary_aggregates_complete_events(tmp_path):
    trace = _trace(tmp_path, [
        _event("GraphReplay", 5), _event("GraphReplay", 7),
        _event("unknown_vendor_op", 3),
    ])
    summary = summarize_trace(trace)
    assert summary["phases"]["graph_replay"]["duration_us"] == 12
    assert summary["phases"]["graph_replay"]["count"] == 2
    assert summary["phases"]["unclassified"]["duration_us"] == 3
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_qwen3_profile_report.py -k 'classify or trace_summary'`

Expected: import failure for the analyzer.

- [ ] **Step 3: Implement deterministic normalization**

The analyzer must process complete (`ph == "X"`) events with numeric `dur`,
retain host/device identity from event args/category/pid/tid, and produce:

```json
{
  "event_count": 0,
  "complete_event_count": 0,
  "total_event_duration_us": 0.0,
  "phases": {"phase": {"count": 0, "duration_us": 0.0, "share": 0.0}},
  "top_events": [{"name": "...", "count": 0, "duration_us": 0.0}],
  "unclassified_names": ["..."]
}
```

Durations are summed event time, not a non-overlapping wall-clock claim. The
report must call this out because concurrent streams can make shares exceed a
wall-clock interpretation. Top events are sorted by duration descending then
name ascending.

- [ ] **Step 4: Merge aggregate benchmark and profile evidence**

Validate the three benchmark JSON files with
`benchmarks.compare_results.load_comparable_results`. Recompute all relative
deltas from raw values. Validate metadata framework names, trace hashes, output
lengths, and workload equality before writing the two normalized files.

- [ ] **Step 5: Run analyzer tests**

Run: `pytest -q tests/test_qwen3_profile_report.py`

Expected: all tests pass, including unknown-event preservation.

- [ ] **Step 6: Commit**

```bash
git add tools/analyze_qwen3_profiles.py tests/test_qwen3_profile_report.py
git commit -m "feat: normalize Qwen3 Chrome traces"
```

### Task 4: Collect the three raw Qwen3 traces on npu2

**Files:**
- Create: `docs/profiling/qwen3/raw/auto-infer.trace.json`
- Create: `docs/profiling/qwen3/raw/omni-npu.trace.json`
- Create: `docs/profiling/qwen3/raw/vllm-ascend.trace.json`
- Create: `docs/profiling/qwen3/manifest.json`
- Create: `docs/profiling/qwen3/summary.json`

**Interfaces:**
- Consumes: profiler launcher and analyzer from Tasks 2–3.
- Produces: final evidence files consumed by the HTML builder.

- [ ] **Step 1: Check device ownership before every run**

Run: `ssh npu2 'npu-smi info'`

Expected: choose one device with no running process. Do not terminate or move
another user's process.

- [ ] **Step 2: Deploy to a new isolated `/data2` directory**

Use `/data2/auto-infer-qwen3-html-report-20260722`. Copy the current worktree
without `.git`, caches, or prior profiler artifacts. Preserve existing model and
framework installations; do not overwrite historical benchmark directories.

- [ ] **Step 3: Capture auto-infer**

Run inside `auto-infer-dev-20260624` with the selected visible device:

```bash
PYTHONPATH=. python benchmarks/profile_qwen3.py \
  benchmarks/comparison_manifest.json auto-infer results/profiling
```

Expected: 16 tokens for every B16 request, valid trace and metadata, no graph
capture failure, no external sampler step.

- [ ] **Step 4: Capture vllm-ascend**

Run the same launcher in the vllm-ascend environment with framework
`vllm-ascend`, no omni plugin variables, and the same visible device.

Expected: valid B16 output, trace, metadata, and digest matching auto-infer.

- [ ] **Step 5: Capture omni-npu**

Run with the documented omni plugin configuration:

```bash
VLLM_PLUGINS=omni-npu,omni_npu_patches \
OMNI_NPU_VLLM_PATCHES=ALL \
PYTHONPATH=. python benchmarks/profile_qwen3.py \
  benchmarks/comparison_manifest.json omni-npu results/profiling \
  --model-compat-slot /data1/models/Qwen3-0.6B
```

Expected: valid output length and trace. Record, but do not hide, digest
difference or plugin warnings.

- [ ] **Step 6: Enforce artifact size and browser format**

Validate JSON and SHA-256. If any trace is larger than 25 MiB, recapture an
eight-token B16 window. Do not truncate JSON or commit compressed-only output.

- [ ] **Step 7: Copy evidence into the worktree and normalize**

Run `tools/analyze_qwen3_profiles.py` with the three remote metadata files and
the retained benchmark JSON files. Confirm generated manifest and summary use
relative raw-trace paths and exact checksums.

- [ ] **Step 8: Commit evidence**

```bash
git add docs/profiling/qwen3
git commit -m "data: add matched Qwen3 profiler evidence"
```

### Task 5: Build the deterministic offline HTML report

**Files:**
- Create: `tools/build_qwen3_architecture_report.py`
- Create: `docs/AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html`
- Modify: `tests/test_qwen3_profile_report.py`

**Interfaces:**
- Consumes: `docs/profiling/qwen3/summary.json`, `manifest.json`, existing
  architecture documents, and current source paths.
- Produces: `build_report(summary: dict, manifest: dict) -> str` and the final
  standalone HTML artifact.

- [ ] **Step 1: Write failing report-content tests**

```python
def test_report_contains_executive_and_engineering_sections(profile_evidence):
    html = build_report(*profile_evidence)
    for anchor in (
        "executive-summary", "matched-benchmark", "profiling-deep-dive",
        "why-faster", "architecture-comparison", "invariants",
        "per-model-artifacts", "acceptance-workflow", "evidence-appendix",
    ):
        assert f'id="{anchor}"' in html


def test_report_links_every_raw_trace_and_labels_evidence(profile_evidence):
    html = build_report(*profile_evidence)
    for framework in ("auto-infer", "omni-npu", "vllm-ascend"):
        assert f'profiling/qwen3/raw/{framework}.trace.json' in html
    assert "实测" in html
    assert "源码观察" in html
    assert "因果推断" in html
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_qwen3_profile_report.py -k report`

Expected: missing report builder.

- [ ] **Step 3: Implement semantic HTML and native visuals**

Generate:

- a sticky table of contents;
- an executive scorecard with exact deltas;
- CSS/SVG metric bars for TTFT, TPOT, throughput, load, allocation, and CV;
- per-framework phase composition and top-event tables from trace summaries;
- a measured/observed/inferred causal-chain table;
- a detailed architecture comparison with advantages and disadvantages;
- a two-column invariant/regenerated matrix;
- a new-model acceptance gate timeline;
- trace download/open instructions, SHA-256 values, reproduction commands, and
  limitations.

All numeric text must be HTML-escaped and formatted by builder helpers. No value
is copied manually from the Markdown performance report when it exists in
normalized JSON.

- [ ] **Step 4: Validate document structure and relative links**

Tests parse HTML with the standard library, assert one `<title>`, one `<h1>`,
all required anchors, local-only assets, no placeholder markers, no missing trace target,
and displayed hashes equal `manifest.json`.

- [ ] **Step 5: Generate and commit**

```bash
python tools/build_qwen3_architecture_report.py
git add tools/build_qwen3_architecture_report.py \
  docs/AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html \
  tests/test_qwen3_profile_report.py
git commit -m "docs: add Qwen3 architecture performance report"
```

### Task 6: Browser, factual, and final publication verification

**Files:**
- Modify only if verification exposes a defect.

**Interfaces:**
- Consumes all prior artifacts.
- Produces a publication-ready report and retained evidence directory.

- [ ] **Step 1: Run complete automated verification**

Run:

```bash
pytest -q
python -m compileall -q auto_infer benchmarks tools tests
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Validate every raw trace**

Run the artifact validator and analyzer against all three files. Confirm each
trace has `traceEvents`, nonzero complete events, a matching SHA-256, and size
below 25 MiB.

- [ ] **Step 3: Render the HTML in Chromium**

Open the local report, inspect desktop and narrow viewport, click every trace
link, and print-preview the report. Capture screenshots of the executive page,
profiling section, architecture table, and invariant/regenerated section for QA;
screenshots are temporary review evidence and are not required deliverables.

- [ ] **Step 4: Independently review facts and claims**

Compare headline numbers to the three retained benchmark JSON files, profile
tables to normalized trace summaries, and architecture statements to current
source. Reject unsupported universal-superiority language or causal claims not
labeled as inference.

- [ ] **Step 5: Final regeneration/idempotence gate**

Run the report builder twice and confirm the second run leaves the HTML byte
identical and `git status --short` clean.

- [ ] **Step 6: Commit any verification corrections**

```bash
git add docs tools tests benchmarks
git commit -m "docs: finalize Qwen3 profiling report evidence"
```
