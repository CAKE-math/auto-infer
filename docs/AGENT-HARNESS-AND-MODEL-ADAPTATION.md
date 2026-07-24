# Agent Harness and Automatic Model Adaptation

## Purpose

The system separates probabilistic Agent work from deterministic inference
runtime work. PIE decides what to investigate and when to retry; auto-infer
proves what a model requires, generates one deployable package, and rejects
unverified configurations.

## Responsibility split

| Layer | Owns | Must not own |
|---|---|---|
| PIE control plane | Resource gathering, investigation, workflow decisions, remote execution, checkpointing, escalation, final report | Runtime introspection logic, package schema, model registration, inference correctness |
| auto-infer Harness | Config and weight-metadata inspection, capability matching, deterministic package generation and validation, structured artifacts | Agent loops, SSH policy, knowledge accumulation, model-specific optimistic guesses |
| auto-infer runtime | Scheduler, KV cache, executor, graph pipeline, kernels, model/backend seams, loading a validated package | PIE imports or workflow state |

The boundary is the public `inferctl` CLI and a StepEnvelope-compatible JSON
result. The deployment handoff is exactly one `model-package.json` plus an
optional package-local Python entrypoint. The runtime never imports PIE.

## Stable workflow

```text
PIE gather / investigate
        |
        v
inferctl capabilities
        |
inferctl inspect model ──> model-manifest.json
        |
capability matcher ──────> supported / partial + exact missing capabilities
        |
inferctl adapt model ────> model-package.json [+ package-local model.py]
        |
inferctl validate package
        |
NPU reference-vs-package token digest
        |
accuracy / stability / performance promotion gates
```

`partial` is a first-class terminal state, not success. It tells PIE precisely
which capability lacks evidence. PIE may add package-local code only when the
existing public model/backend seam can express it. A new reusable runtime
capability is a separate framework change with its own review.

## Public commands

```bash
inferctl capabilities
inferctl inspect model MODEL --artifacts ARTIFACT_DIR
inferctl adapt model MODEL --output PACKAGE_DIR --artifacts ARTIFACT_DIR
inferctl validate package PACKAGE_DIR --model MODEL --artifacts ARTIFACT_DIR
```

All results include `status`, `step_id`, `created_at`, `error_summary`,
`artifacts`, `provenance`, and `result`. Exit code `0` means `ok` or `skipped`,
`2` means `partial`, and `1` means `failed`.

Inspection reads `config.json`, the safetensors index, or safetensors headers;
it does not load model tensors. Matching is based on architectural evidence,
not model names. Current automatic templates cover proven GQA/Qwen-like and
MLA+MoE/DeepSeek-like BF16 layouts. Quantization has a reserved manifest
interface but is intentionally disabled.

## Model package contract

The package pins:

- schema version and source config SHA-256;
- source architecture to runtime entrypoint mapping;
- attention family and feature evidence;
- BF16 execution policy;
- disabled/reserved quantization policy;
- deterministic validation state.

Package paths are relative and cannot escape the package directory. At runtime,
`ModelRegistry.register_package` validates the package before loading its
entrypoint. A conflicting registration is rejected.

## NPU promotion gate

Structural validation alone is insufficient. Run the same greedy token IDs
through a trusted built-in model and the candidate package:

```bash
python scripts/verify_model_package.py \
  --reference-model /models/reference \
  --candidate-model /models/candidate \
  --package /artifacts/model-package \
  --prompt-ids '[[1, 2, 3], [100, 200]]' \
  --max-tokens 32 \
  --mode paged
```

The tool loads the models sequentially, releases the first executor, and
compares exact per-request output tokens plus stable SHA-256 digests. Promotion
requires:

1. `inferctl validate package` succeeds.
2. NPU load and smoke generation succeed.
3. Reference and package outputs match exactly in deterministic greedy mode.
4. The project accuracy suite passes.
5. Stability and performance tests pass without weakening the first four
   gates.

## What remains stable and what varies

Stable across models:

- Harness result schema, package schema, exit semantics, ownership boundary;
- runtime model/backend registry seams;
- BF16 correctness gates and source-drift validation;
- NPU promotion sequence.

Regenerated for every model:

- inspected config/weight manifest and source digest;
- capability evidence and missing-capability list;
- architecture mapping and optional package-local entrypoint;
- token-digest, accuracy, stability, and performance artifacts.
