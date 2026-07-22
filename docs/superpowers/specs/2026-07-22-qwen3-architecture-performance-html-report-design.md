# Qwen3 Architecture and Performance HTML Report Design

Date: 2026-07-22

## Objective

Produce one management-readable and engineering-auditable HTML report that
explains the current auto-infer architecture, compares it with omni-npu and
vllm-ascend, and grounds the Qwen3 performance explanation in matched benchmark
results plus bounded raw profiler traces.

The report must answer four decisions:

1. what auto-infer measurably wins and under which workload;
2. which architectural mechanisms plausibly cause that result;
3. where the other frameworks remain stronger;
4. which contracts are framework invariants versus model-specific artifacts
   that must be regenerated and revalidated for every checkpoint family.

## Deliverables

- `docs/AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html`: a self-contained,
  responsive, print-friendly Chinese report with English metric and operator
  names where they improve precision.
- `docs/profiling/qwen3/manifest.json`: workload, software versions, hardware,
  profiler windows, file sizes, checksums, and pass/fail identity metadata.
- `docs/profiling/qwen3/summary.json`: normalized cross-framework metrics and
  trace-derived phase/kernel summaries consumed by the report.
- `docs/profiling/qwen3/raw/auto-infer.trace.json`
- `docs/profiling/qwen3/raw/omni-npu.trace.json`
- `docs/profiling/qwen3/raw/vllm-ascend.trace.json`

The three raw files use the Chrome Trace Event format and must open locally in
Perfetto (`https://ui.perfetto.dev`) or legacy `chrome://tracing`. They are
bounded steady-state captures, not entire model-load traces. The HTML lists
their relative paths, SHA-256 hashes, sizes, capture intervals, and opening
instructions. A trace that exceeds the repository's safe file-size limit must
be recaptured with a shorter active window; compression alone is not accepted
because the requested artifact must remain directly openable.

## Evidence Model

The report labels every statement as one of:

- **Measured**: derived from the matched Qwen3 benchmark JSON or raw trace;
- **Observed in source**: an architectural fact tied to concrete modules and
  control flow;
- **Inference**: a causal interpretation consistent with measurements and
  source, but not independently proven by an ablation.

The main benchmark remains the existing matched workload: Qwen3-0.6B, one
Ascend 910B1, BF16, greedy/ignore-EOS, B1 latency, B16 throughput, 128 generated
tokens, one warm-up, twenty measurements, and 14,464 usable KV tokens in every
framework. Existing aggregate JSON remains authoritative for headline numbers.

Profiler collection adds a second, explicitly separate workload phase. Each
framework runs sequentially on the same idle device and captures the smallest
repeatable steady-state region that contains prefill plus multiple decode
steps. Profiler overhead results are never substituted for unprofiled
throughput, TTFT, or TPOT.

Correctness metadata records output length and digest. auto-infer and
vllm-ascend may be described as token-identical for the validated digest;
omni-npu is performance-comparable only because its digest differs.

## Profiler Contract

Each framework-specific launcher must emit the same normalized metadata:

- framework/version and exact source revision;
- model path and checkpoint/config digest;
- visible NPU, CANN, torch, torch-npu, vLLM/plugin versions;
- prompt token count, output token count, batch size, KV capacity, dtype,
  sampling mode, warm-up count, and capture schedule;
- raw trace path, size, SHA-256, start/end timestamps;
- output digest and generated length;
- operator count, device duration, host launch duration, stream occupancy proxy,
  top operators by total/self time, and categorized phase totals.

The normalized phase taxonomy is intentionally small:

1. host scheduling and batch preparation;
2. host-to-device metadata/input updates;
3. graph compile/capture or replay/launch;
4. attention and KV write;
5. dense projections, MLP, and normalization;
6. LM head and sampling;
7. device-to-host output materialization;
8. synchronization/idle/unclassified time.

Mappings from raw operator names to these categories are versioned in the
manifest. Unknown operators remain visible as `unclassified`; they are not
silently assigned to a favorable bucket. Counts and durations are reported
both in absolute units and as percentages of the captured device/host window.

## HTML Information Architecture

The page uses an executive-first, engineering-deep structure:

1. **Executive decision panel** — verified wins, bounded conclusion, key risks,
   and three recommended investment priorities.
2. **Matched benchmark scorecard** — TTFT, TPOT, B16 throughput, load time,
   equal-capacity peak allocation, CV, digest status, and exact relative deltas.
3. **Qwen3 profiling deep dive** — per-framework timeline summary, phase
   composition, top operators, kernel/launch counts, synchronization evidence,
   and direct raw-trace links.
4. **Why auto-infer is faster** — a measured/observed/inferred causal chain for
   persistent graph replay, side-stream graph-task metadata update, double
   buffering, persistent staging and dirty rows, packed QKV/gate-up, captured
   BF16 head/argmax, and reduced orchestration depth.
5. **Architecture comparison** — control flow, state ownership, extension seam,
   graph lifecycle, scheduler, serving, distributed/MoE, model breadth,
   quantization breadth, operational maturity, and maintenance cost.
6. **Advantages, disadvantages, and scope boundary** — explicitly retain
   vllm-ascend's ecosystem/API maturity and omni-npu's optimized model/operator
   breadth as competitor strengths.
7. **Invariant versus regenerated contract** — the architectural boundary for
   future model onboarding.
8. **New-model acceptance workflow** — checkpoint inspection, generated
   artifacts, precision gates, profiling, stability, and release evidence.
9. **Evidence appendix** — formulas, raw samples, versions, file hashes,
   limitations, source references, and reproduction commands.

Visuals are native HTML/CSS/SVG with no CDN or runtime dependency. Tables retain
their values in text for copy/paste and printing. Color is not the sole carrier
of meaning, and every chart has a textual interpretation.

## Architectural Invariants

The report identifies the following as not model-generated and not negotiable
without an explicit architecture revision:

- `EngineCore -> BatchPlan -> Executor -> ExecutionResult` protocol;
- single ownership of request, scheduler, KV, and completion state;
- model-declared attention family resolved through registries;
- separately registered recurrent-MTP capability with fail-fast unsupported
  behavior;
- one graph-FIA capture/update lifecycle and event-ordering contract;
- persistent, fixed-address staging with dirty block-table updates;
- precision-first release gates and matched benchmark methodology;
- no model-specific branches in the engine/scheduler composition root;
- explicit experimental boundaries for unwired P/D and unsupported MLA MTP;
- benchmark output schema, raw evidence retention, and correctness-before-speed
  ordering.

These can evolve only through a versioned framework design, cross-model
regression, and new baseline—not because one checkpoint needs a special case.

## Per-Model Regenerated Artifacts

Every model/checkpoint family must regenerate or revalidate:

- checkpoint/config/weight-name inventory and model-family adapter;
- local TP/EP head and expert geometry;
- attention dimensions, RoPE variant/tables, cache layout, block size, KV budget,
  maximum sequence length, and supported FIA constraints;
- packed/fused QKV and gate-up weights plus any dtype/quantization metadata;
- graph gear ladder, scratch-block budget, capture-success matrix, graph handles,
  and memory envelope;
- MTP layer count, prefix/geometry, recurrent depth boundary, acceptance by
  position, and capability registration;
- sampling/head precision policy and BF16/FP32 parity threshold;
- golden prompts, logits/token digests, eager/paged/graph identity, and
  framework/reference accuracy baselines;
- unprofiled performance baseline, raw profiler traces, operator-category map,
  stability distribution, and regression thresholds;
- distributed topology validation for TP/EP/SP/CP combinations actually
  claimed for that model.

Generated artifacts are keyed by model/config/weights digest plus software and
hardware version. They must never be reused solely because two models share an
architecture class name.

## Validation and Publication Gates

- Recompute every percentage and ratio from normalized JSON; no hand-entered
  derived metric appears only in HTML.
- Validate trace JSON syntax and required Chrome trace keys; open each trace in
  a Chromium-compatible viewer or run a parser acceptance check.
- Confirm all relative HTML links resolve and all displayed trace hashes match.
- Verify the page at desktop and narrow widths and in print/PDF layout.
- Run the complete host test suite and tracked whitespace checks.
- Independently review factual claims against raw benchmark results, trace
  summaries, and current source architecture.
- Do not publish a universal superiority claim: conclusions remain scoped to
  the measured Qwen3 workload and reviewed core.

## Non-goals

- Changing runtime behavior or optimizing a framework during profiling.
- Treating profiled timings as headline production throughput.
- Claiming omni-npu token identity where the recorded digest differs.
- Embedding multi-hundred-megabyte traces inside the HTML document.
- Expanding P/D, MLA MTP, quantization, serving, or model support as part of the
  report task.
