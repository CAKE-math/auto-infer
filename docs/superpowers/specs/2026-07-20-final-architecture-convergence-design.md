# Final Architecture Convergence Design

Date: 2026-07-20
Target: host tests plus one Ascend 910B1 on `npu2`

## Goal

Make auto-infer's supported execution surface smaller, explicit, and easier to
extend than the corresponding vllm-ascend and omni-npu paths without changing
token results, KV ownership, graph addresses, or benchmark policy.

## Constraints

- Preserve the public `EngineConfig`, `LLM`, `EngineService`, and executor behavior.
- Preserve sync, async, preemption, continuous batching, graph decode, and MiMo
  K=1 MTP correctness before removing compatibility code.
- Do not add dependencies or generality without a current production consumer.
- Host tests protect control flow; NPU2 protects graph capture, parity, stability,
  and performance.
- A compatibility path may be deleted only after every production caller is gone
  and its replacement passes the relevant NPU2 gate.

## Architecture

### Execution lifecycle

`EngineCore` remains the sole owner of mutable request and scheduler state. A
small internal lifecycle API owns scheduling with preemption, result validation,
token emission, request completion, and cleanup. Sync and MTP policies provide
only their different execute/commit semantics; async retains its queue-specific
optimistic commit but uses the same completion and cleanup primitives.

Execution backends are described by immutable backend specifications registered
under the four supported mode names. Each specification validates its compatible
configuration, derives constructor arguments, and loads its executor class. The
factory no longer duplicates a central mode switch.

### Attention construction

Attention families register one builder each. `build_attention_backend` validates
the requested mode, resolves the family builder, and allocates caches. Adding a
family does not require editing the dispatcher.

### MTP pipeline

Confirmed-row layout is a speculative-decoding value object, not a worker detail.
It moves to `spec_decode/layout.py`; the runner and stager depend on that module.
Shared pinned-host-buffer and dirty-span logic moves to one staging utility.

The two-stage target/drafter graph pipeline is the only normal MTP graph path.
Startup capture failures are explicit initialization failures after NPU2 proves
all supported gears capture. The old fused graph classes, capture body, replay,
and fallback counters are then removed from the production runner.

Eager prefill remains because it handles prompt processing rather than duplicating
speculative graph decode.

### Optional and compatibility surfaces

Mooncake is not advertised as active until a connector constructs it and its
receive contract is executable. The unused half-implementation is removed; the
existing HCCL connector remains the supported P/D transport.

Test instrumentation stays in tests. Legacy parity implementations move to tools
or are deleted after the parity harness uses the unified production forward.
Backward-compatible public facades remain only when a real external or documented
consumer exists.

## Error handling

- Unsupported backend/family names fail at lookup with the registered choices.
- Backend-specific invalid configurations fail before model loading.
- Missing executor output continues to raise `EngineStalledError`.
- MTP graph capture failure names the exact target/drafter gear and aborts startup;
  it never silently changes the execution algorithm.
- Request cleanup is idempotent and centralized.

## Verification

1. Host unit tests cover registries, shared lifecycle helpers, layout, staging,
   continuous batching, preemption, async backfill, and absence of legacy symbols.
2. `compileall` and pyflakes must report no unexplained findings; side-effect NPU
   imports are marked explicitly.
3. NPU2 runs Qwen graph parity/stability and Moonlight plain/MTP parity at B1/B4/B16.
4. NPU2 runs repeated continuous-batching and capture-gear coverage tests.
5. Performance is accepted only if the architecture cleanup does not regress the
   retained vllm-ascend comparison outside normal run variance.

## Acceptance

- One production implementation for each supported behavior.
- No production definition without a production caller, public contract, or
  documented side-effect responsibility.
- Dependency direction is engine -> executor -> runner -> staging/spec primitives.
- Adding an execution backend or attention family is registration-only.
- All host and NPU2 gates pass with retained logs.
