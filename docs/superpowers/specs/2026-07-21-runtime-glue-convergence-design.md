# Runtime Glue Convergence Design

Date: 2026-07-21

## Goal

Remove the remaining provable redundancy and ownership drift around the
production decode pipeline without broadening supported behavior or changing
the validated BF16 token stream.

## Scope

The convergence keeps two deliberate extension surfaces:

- `auto_infer.pd` remains a low-level P/D KV-transfer interface. It is not
  described as a complete serving lifecycle until a production request/KV
  topology owns it.
- MLA MTP remains an explicit attention-family capability seam. This pass does
  not claim or implement DeepSeek/MLA MTP execution; an MLA request must fail
  capability validation with a precise unsupported error.

Everything else follows deletion-first YAGNI: code reachable only from tests or
verification scripts moves out of the installed runtime or is deleted.

## Staging Ownership

`worker/staging.py` owns two shared staging primitives:

1. device-token splicing from `DeviceTokenRef` owners into persistent input
   buffers;
2. dirty block-table upload, including shadow update and both copied-row and
   copied-element accounting.

Decode, continuation, speculative-decode, prefill, and MTP staging call this
single helper. The helper returns the copied row and element counts so each
stager retains its existing public counters without sharing mutable global
state. This removes the five hand-written loops and fixes the continuation
counter drift.

## MTP Capability Seam

Attention-family registration gains an optional MTP backend builder separate
from the normal model-layer backend builder. Workers request an MTP backend by
family and mode; they never import `GqaFIABackend` or `GraphGqaBackend`
directly.

The GQA family implements paged and graph MTP construction with the requested
MTP layer prefix and one-layer cache topology. The MLA family registers no MTP
implementation in this pass. Requests against an unsupported family fail
before runner capture with an error naming the family and mode. This preserves
the MLA MTP extension point without pretending that its model weights,
projection contract, or NPU token parity have been validated.

## Graph Attention Lifecycle

GQA and MLA keep their projection and cache-layout math separate. Their common
graph-task state machine moves to one internal mixin/helper that owns:

- capture-mode begin/end;
- graph-task entry storage;
- replay/update iteration and stream selection.

Backend-specific code supplies the FIA invocation and cache view. The refactor
must not add a Python branch, allocation, synchronization, or kernel to the
captured path. Existing graph-task event ordering remains unchanged.

## Runtime Cleanup

- Remove `serving/router.py`; it is unreachable from supported serving entry
  points. Its tests and obsolete verification script are removed with it.
- Move the SSE response parser from `auto_infer.serving` to `benchmarks`, where
  its only non-test consumer lives. Verification tools import the benchmark
  helper explicitly.
- Move W8A8 MoE numerical-probe code out of `layers/moe/fused_moe.py` and into
  its verification script. The runtime retains the existing quantization
  interface but does not advertise an unwired MoE W8A8 path.
- Remove the retired `ep_all_reduce` wrapper from distributed runtime;
  comparison scripts call `torch.distributed.all_reduce` locally.
- Replace the stateless `DecodeEpilogue` class with the module function
  `is_capturable_greedy`.
- Remove the MLA self-import and duplicate local imports.
- Keep `pd/connector.py`, add an explicit experimental low-level contract, and
  correct reports that describe it as an integrated production topology.

## Bootstrap Ownership

The engine composition root initializes the distributed mesh exactly once
before model/executor construction. `EngineCore` consumes an already-validated
runtime and does not create process groups. Test injection remains possible
without distributed side effects. Idempotence stays in `parallel_state` as a
safety property, not as a substitute for unique ownership.

## Sampling Calls

`build_sampling_tensors` and `sample_batched` are already the single shared
implementations. Their four call sites are not treated as four algorithm
copies. This pass may centralize imports or introduce a small request-sampling
function only when it removes code without hiding distinct row-selection or
captured-output behavior.

## Validation

Each behavioral change follows a red/green test:

- continuation dirty upload must count rows and elements;
- every stager must use the shared upload primitive;
- worker MTP modules must have no concrete GQA imports;
- GQA MTP builder returns the requested mode and MLA reports unsupported;
- supported entry points must not import the removed router or runtime SSE
  helper;
- distributed initialization must have one production caller;
- graph GQA/MLA lifecycle tests must retain capture/update ordering.

Final gates are the complete host suite, compileall, pyflakes, import-boundary
and duplicate-body scans, tracked whitespace, focused BF16 packed-MLA parity on
npu2, and the retained MiMo graph-MTP K1/K2 token-identity smoke. Performance
must not regress outside existing run-to-run noise.

## Non-goals

- Implementing or claiming DeepSeek/MLA MTP execution.
- Integrating P/D into serving or adding a disaggregated deployment topology.
- Adding quantization modes beyond the existing public interface.
- Expanding router, replica, model, or serving feature breadth.
