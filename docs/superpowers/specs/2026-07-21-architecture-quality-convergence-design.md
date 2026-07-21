# Architecture Quality Convergence Design

## Goal

Converge auto-infer so its supported scope has one clear dependency direction,
one execution contract per behavior, no unwired production reference code, and
correct packed MLA evaluation.

## Scope

This change fixes the findings from the 2026-07-21 architecture review. It does
not redesign serving, change model mathematics, add compatibility with vLLM, or
expand the supported model and quantization matrix.

## Dependency Direction

The intended dependency flow is:

`config/model contracts -> neutral runtime contracts -> engine/worker composition`

Two neutral modules own contracts currently placed too high in the tree:

- `auto_infer/executor_backends.py` owns executor backend specifications,
  registration, validation, constructor arguments, and lazy implementation
  loading. Both configuration and the engine factory may depend on it.
- `auto_infer/graph_tasks.py` owns ACL graph-task entries and capture/update
  primitives. Attention backends and the worker-side replay pipeline may depend
  on it; attention layers must not import workers.

`ExpertParallelTopology` moves to `auto_infer/distributed/topology.py`, so the
distributed runtime no longer imports an MoE layer merely to return its cached
communication resources.

## Packed MLA Correctness

Dense attention must interpret `ForwardContext.cu_seqlens_q` as cumulative
request boundaries. `MlaDenseBackend` will calculate causal attention once per
segment and concatenate the outputs, matching the existing GQA dense contract.
Tests compare packed execution against independently evaluated sequences and
must fail against the current implementation before production code changes.

The implementation may share a small dense-attention core between GQA and MLA
only if doing so reduces duplication without adding branching to paged or graph
hot paths.

## MTP Geometry

`num_speculative_tokens` is the requested recurrent proposal depth. A checkpoint
with one trained MTP layer may recurrently reuse that layer for any positive
depth in eager/paged execution. Graph execution separately enforces the retained
NPU token-parity boundary of two draft tokens; K3 remains unavailable until its
four-query-row verification path passes the same token-identity gate.

Both paged and graph executors derive geometry from checkpoint weights. Graph
execution additionally requires that the target query width (`draft_depth + 1`)
fit inside one block and that the requested depth pass its NPU-verified boundary.
Model layer prefixes come from `MtpGeometry.layer_prefix()` rather than string
literals.

`MtpDrafter` receives a fully constructed `RecurrentMtpHead` through its normal
initializer. A `from_model()` factory performs model-specific backend setup;
there is no `__new__` bypass.

## Redundancy Removal

- Remove the unwired profiling implementation.
- Move the retired all-reduce MoE numerical reference out of the installed
  `auto_infer` package and into the EP verification script.
- Remove the tests-only `engine.errors` compatibility module and import public
  errors from `auto_infer.errors`.
- Remove the tests-only `validate_execution_config` wrapper; backend argument
  resolution remains the canonical validation path.
- Remove `mesh_axis_groups`; topology tests use `ParallelMesh.groups()`.
- Keep model, attention, and executor registries as real extension seams, but
  reject duplicate names unless replacement is explicitly requested. Built-in
  model registration uses the same public operation, so it is not dead code.
- Stop re-exporting `StatLogger` through serving metrics.

## Documentation

Architecture comparison documents will regenerate current physical source
counts and replace unqualified superiority claims with auditable statements:
smaller review surface, explicit composition, zero vLLM imports, and lower
adaptation cost inside the supported scope. Feature breadth and production
maturity remain separate dimensions.

## Verification

Each behavioral change follows red-green-refactor:

1. Add a failing packed-MLA isolation test, then fix the backend.
2. Add failing dependency-boundary tests, then move neutral contracts.
3. Add failing MTP depth/constructor tests, then unify geometry and construction.
4. Add failing registry duplicate tests, then harden registration.
5. Remove redundant code and update imports while keeping focused suites green.

Final acceptance requires the complete host test suite, `compileall`, import-SCC
inspection, repository whitespace validation, a clean worktree, and a focused
NPU packed-MLA parity test when the NPU environment is available.

## Non-Goals

- No serving feature expansion or protocol compatibility layer.
- No runner rewrite solely to reduce line count.
- No quantization implementation.
- No deletion of tested production capabilities merely because they are not on
  the current benchmark path.
