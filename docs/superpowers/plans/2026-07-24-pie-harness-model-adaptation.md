# PIE + auto-infer Harness and Model Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic auto-infer Harness and a PIE control-plane skill that can inspect, adapt, validate, and load new compatible model packages without modifying the stable runtime.

**Architecture:** PIE performs non-deterministic investigation and iteration. `inferctl` owns checkpoint facts, capability matching, package generation, validation, structured artifacts, and runtime registration. The only shared state is StepEnvelope-compatible JSON plus one `model-package.json`.

**Tech Stack:** Python 3.11+, argparse, dataclasses, JSON, hashlib, importlib, pytest, PIE Markdown skills.

## Global Constraints

- The production runtime must not import PIE.
- PIE must invoke public `inferctl` commands and must not import auto-infer internals.
- `model-package.json` is the only runtime adaptation descriptor.
- Harness exit codes are `0=ok`, `2=partial`, `1=failed`.
- Capability uncertainty returns `partial`; it never generates a package marked runnable.
- Generated packages default to BF16 and retain a disabled quantization interface.
- EngineCore, scheduler, KV cache, serving, and graph runners receive no model branches.
- Existing untracked files in either repository are user-owned and must remain untouched.

---

### Task 1: Harness envelope and artifact writer

**Files:**
- Create: `auto_infer/harness/__init__.py`
- Create: `auto_infer/harness/artifacts.py`
- Create: `tests/test_harness_artifacts.py`

**Interfaces:**
- Produces: `HarnessResult(status, step_id, result, artifacts, error_summary)`
- Produces: `write_result(result: HarnessResult, path: Path) -> None`
- Produces: `exit_code(status: str) -> int`

- [ ] **Step 1: Write failing envelope tests**

Cover StepEnvelope base keys, UTC timestamp format, provenance with framework
commit/config SHA, stable JSON formatting, and status-to-exit-code mapping.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_harness_artifacts.py
```

Expected: import failure because `auto_infer.harness.artifacts` does not exist.

- [ ] **Step 3: Implement the artifact primitives**

Use a frozen dataclass, UTC ISO-8601 timestamps with `Z`, sorted/indented JSON,
and subprocess-free git revision lookup from `.git` only when available.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest -q tests/test_harness_artifacts.py
```

Expected: all tests pass.

---

### Task 2: Checkpoint inspection and capability matching

**Files:**
- Create: `auto_infer/harness/inspect.py`
- Create: `auto_infer/harness/capabilities.py`
- Create: `tests/test_harness_inspect.py`

**Interfaces:**
- Produces: `inspect_model(model_path: Path) -> dict`
- Produces: `match_capabilities(manifest: dict) -> dict`
- Consumes: `config.json`, optional `model.safetensors.index.json`

- [ ] **Step 1: Write failing model-fixture tests**

Create temporary fixtures for Qwen2-like GQA, Qwen3-like explicit head
dimension/QK norm, DeepSeek-like MLA+MoE, unknown attention, missing required
fields, and absent weight evidence. Assert exact attention/features/cache
facts and supported/partial verdicts.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_harness_inspect.py
```

Expected: import failure for the missing inspector.

- [ ] **Step 3: Implement inspection**

Read config once, fingerprint its bytes, derive attention without model-name
branching, inspect weight-index keys without tensor reads, and emit sorted
required/missing capability lists.

- [ ] **Step 4: Verify GREEN**

Run the Task 2 test file and Task 1 regression.

---

### Task 3: Model-package generation, validation, and registration

**Files:**
- Create: `auto_infer/harness/package.py`
- Modify: `auto_infer/models/registry.py`
- Modify: `auto_infer/config/__init__.py`
- Modify: `auto_infer/engine/factory.py`
- Modify: `auto_infer/entrypoints/cli.py`
- Create: `tests/test_model_package.py`
- Modify: `tests/test_executor_factory.py`
- Modify: `tests/test_serving_cli.py`
- Modify: `tests/test_architecture_convergence.py`

**Interfaces:**
- Produces: `generate_package(manifest: dict, output_dir: Path) -> dict`
- Produces: `validate_package(package_dir: Path, model_path: Path) -> dict`
- Produces: `register_package(package_dir: str, model_path: str) -> None`
- Adds: `ModelConfig.model_package: str | None`
- Adds: `auto-infer serve --model-package PATH`

- [ ] **Step 1: Write failing package tests**

Assert deterministic GQA/Qwen3-like/MLA entrypoint selection, BF16 policy,
reserved quantization interface, rejection of partial manifests, config
fingerprint drift, relative custom entrypoint loading, duplicate registration,
and composition-root registration before executor construction.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_model_package.py tests/test_executor_factory.py tests/test_serving_cli.py tests/test_architecture_convergence.py
```

Expected: failures for missing package APIs and CLI/config fields.

- [ ] **Step 3: Implement package lifecycle**

Generate only from a supported capability verdict. Validate schema version,
source SHA, architecture list, entrypoint, and model contract. Support
`module:Class` and package-relative `./file.py:Class`. Register before the
executor backend loads.

- [ ] **Step 4: Verify GREEN**

Run the Task 3 test set and ensure architecture convergence still proves no
engine/runner model branch.

---

### Task 4: `inferctl` public CLI

**Files:**
- Create: `auto_infer/harness/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/test_inferctl.py`

**Interfaces:**
- Adds executable: `inferctl`
- Adds commands: `capabilities`, `inspect model`, `adapt model`,
  `validate package`

- [ ] **Step 1: Write failing CLI tests**

Invoke `main(argv)` for all commands. Assert fixed artifact names, JSON on
stdout, non-interactive behavior, exit codes 0/2/1, unchanged checkpoint
contents, and failed validation diagnostics.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/test_inferctl.py
```

Expected: import failure for `auto_infer.harness.cli`.

- [ ] **Step 3: Implement thin command handlers**

Handlers compose Tasks 1–3 and contain no inspection, matching, or package
logic. `--artifacts` defaults to `.inferctl-artifacts`; commands write
`inspect-model.json`, `adapt-model.json`, or `validate-package.json`.

- [ ] **Step 4: Verify GREEN**

Run Task 4 tests, all Harness tests, `python -m compileall -q auto_infer`, and
the full auto-infer suite.

---

### Task 5: PIE `adapt-auto-infer` control plane

**Files in `/Users/peterzheng/Code/Python/vllm-workspace/PIE`:**
- Create: `skills/adapt-auto-infer/SKILL.md`
- Create: `skills/adapt-auto-infer/ci.yaml`
- Create: `skills/adapt-auto-infer/references/harness-contract.md`
- Create: `knowledge/frameworks/npu/auto-infer/_index.md`
- Create: `knowledge/frameworks/npu/auto-infer/common.md`
- Modify: `knowledge/frameworks/npu/_index.md`
- Modify: `knowledge/KNOWLEDGE-MAP.md`
- Modify: `skills/model-orchestrator/SKILL.md`
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `README_EN.md`
- Create: `tests/test_adapt_auto_infer_contract.py`

**Interfaces:**
- Consumes: public `inferctl` commands and StepEnvelope-compatible results
- Produces: `.ci-state/<model>/adapt-auto-infer.json`
- Adds orchestrator input: `target_framework: auto-infer | vllm-ascend`

- [ ] **Step 1: Write failing PIE repository tests**

Assert skill metadata declares auto-infer/Ascend support, every workflow
invocation uses `inferctl`, checkpoint protocol is present, stable-core edits
are prohibited, result fields are StepEnvelope-compatible, framework knowledge
is routable, and the orchestrator dispatches by `target_framework`.

- [ ] **Step 2: Verify RED**

Run from PIE:

```bash
pytest -q tests/test_adapt_auto_infer_contract.py
```

Expected: failures because the skill and routing do not exist.

- [ ] **Step 3: Implement the PIE skill**

Define inspect → adapt/Agent gap resolution → static validate → remote runtime
validate → final envelope. PIE may edit candidate package files; reusable
runtime capability work is an explicit escalation and separate auto-infer
change.

- [ ] **Step 4: Verify GREEN**

Run the PIE test, knowledge-link checker, and existing envelope validation
examples. Do not modify `contract/`, which is generated externally.

---

### Task 6: Cross-repository and NPU acceptance

**Files:**
- Modify: `docs/AGENT-HARNESS-AND-MODEL-ADAPTATION.md`
- Create: `scripts/verify_model_package.py`
- Create: `tests/test_verify_model_package.py`

**Interfaces:**
- Consumes: a generated package plus a model checkpoint
- Produces: built-in/package output digests and equality verdict

- [ ] **Step 1: Add a host-testable verification script test**

Assert argument parsing, deterministic digest construction, and structured
failure when outputs differ.

- [ ] **Step 2: Implement the verification script and operator document**

Document responsibilities, commands, artifact layout, promotion gates, and
the unsupported-capability escalation path.

- [ ] **Step 3: Run final local verification**

Run:

```bash
python -m compileall -q auto_infer scripts
pytest -q
```

Run PIE tests and `python scripts/check-knowledge-links.py`.

- [ ] **Step 4: Run npu2 acceptance**

Generate a package that aliases a new architecture name to the Qwen3-compatible
implementation, point a copied config at that architecture without copying
weights, and run built-in versus package BF16 greedy generation. Require equal
token digest and a passing static package artifact.

- [ ] **Step 5: Commit each repository**

Commit auto-infer implementation and PIE plugin changes independently. Preserve
PIE's pre-existing untracked `.claude-plugin/marketplace.json`.

