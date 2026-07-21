# Auto-Infer Architecture Convergence Design

**Date:** 2026-07-19

## Objective

Converge auto-infer from a collection of working inference paths into one
coherent framework architecture whose internal contracts are simpler and more
independent than omni-npu and vllm-ascend. Phase one is accepted on architecture,
correctness, and stability. Phase two compares all three frameworks under the
same NPU workload.

The project is version 0.0.1, so internal APIs may change. The compatibility
boundary is `LLM.generate` and the externally visible OpenAI HTTP semantics.

## Non-goals

- Phase one does not claim a performance win over omni-npu or vllm-ascend.
- This work does not add unrelated model families or new quantization formats.
- Existing `/data2/auto-infer-eval*` directories on npu2 are not overwritten.
- Model mathematics are not rewritten unless required to satisfy the unified
  execution contract.

## Architectural Invariants

1. `EngineConfig` is the only runtime configuration source.
2. EngineCore schedules requests but does not expose mutable Scheduler state to
   an execution backend.
3. Every execution mode consumes `BatchPlan` and produces `ExecutionResult`.
4. Eager, paged, graph, and MTP behavior are runner strategies behind one
   executor adapter, not separate model-loading stacks.
5. Models describe model mathematics and architecture metadata. Attention
   backend selection and construction belong to a backend registry.
6. TP, DP, EP, CP, and SP are explicit axes of one validated parallel mesh.
7. A serving engine owns one persistent EngineCore and one persistent physical
   executor. Thread and process transports preserve that ownership model.
8. Every asynchronous response is demultiplexed by request id.
9. Unschedulable work terminates with a structured error; it never spins without
   progress.
10. Every documented capability points to an existing automated test or
    executable verification script.

## Configuration and Construction

`EngineConfig` owns model, cache, scheduler, execution, parallel, speculative
decoding, and observability settings. Its validation rejects:

- non-positive capacities and batch limits;
- cache capacity insufficient for any admitted request;
- graph/spec/async combinations unsupported by the selected runner;
- a parallel mesh whose product or axis mapping disagrees with `WORLD_SIZE`;
- inconsistent model length, block size, runner capacity, or dtype settings.

One `build_executor(config)` path loads the model and selects a runner. API,
evaluation, IPC, and Python entrypoints do not repeat executor parameters.

## Execution Contract

The scheduler produces immutable request views inside a `BatchPlan`. A request
view contains the ids and metadata required for execution, including token ids,
computed-token count, query length, block table, sampling parameters, and
prefill/decode role. It does not expose Scheduler methods or mutable request
containers.

The executor returns `ExecutionResult`, containing sampled or emitted tokens,
optional next drafts, per-request errors, and execution statistics. EngineCore is
the sole owner of logical request-state transitions.

The common executor owns model loading, synchronous execution, sampled-token
threading, asynchronous collection, and shutdown. Runner strategies own only
device input preparation and device execution. A runner advertises capabilities
such as graph capture, asynchronous collection, and MTP; invalid configurations
fail at construction time.

## Model and Attention Boundaries

`BaseCausalLM` retains the shared decoder forward skeleton for supported decoder
families. Its contract is reduced to model-specific mathematics, weight loading,
and architecture metadata. Backend factories move out of model subclasses.

The attention registry selects a dense, paged FIA, or graph implementation from
architecture metadata plus execution mode. Adding a backend must not modify a
model. Adding a model that uses an existing attention family must not modify an
engine or runner. Contract tests enforce both extension directions.

This contract deliberately describes the currently supported decoder scope; it
does not claim that hybrid, multimodal, or state-space models require only the
same hooks.

## Parallel Mesh

`ParallelMesh` represents TP, DP, EP, CP, and SP axes explicitly and computes
rank coordinates and groups deterministically. It validates world size before
creating HCCL groups. Each collective uses its named axis; CP never aliases TP
implicitly, and configured DP is never inferred as `WORLD_SIZE / TP` unless that
is the validated mesh definition.

Host tests cover group coverage, disjointness, coordinates, invalid products,
and combinations. npu2 tests cover real HCCL initialization and representative
TP2 and EP2/SP2 collectives.

## Serving and IPC

Both synchronous and asyncio frontends submit to a `RequestBroker`. The broker
owns request queues, response streams, cancellation, backpressure, and lifecycle
state. A persistent `EngineService` owns the EngineCore and drives steps.

The in-process transport uses thread-safe queues. The process transport uses
request and response envelopes containing request id and message kind. A single
response-demultiplexing loop routes messages to per-request queues; request
consumers never call a shared response queue directly.

The process worker creates EngineCore once, accepts multiple requests, and
continuously batches them. Consequently prefix-cache affinity in the router is
real. Router load tracks live requests and is decremented on completion or
cancellation. Prefix-affinity metadata is bounded and evictable.

All services expose `close()`. Shutdown rejects new submissions, terminates or
cancels active requests, joins threads/processes, and closes executors. A request
failure is isolated. An executor failure fails all affected requests and rebuilds
the executor only through its explicit recovery contract.

## Error and Progress Model

Each EngineCore step must either execute work, complete/fail/cancel a request, or
report that it is waiting on an in-flight asynchronous result. If none applies
while unfinished requests exist, the engine raises a no-progress error with the
blocking request and required capacity.

Oversized prompts, duplicate request ids, invalid sampling parameters, closed
services, incompatible execution modes, and malformed IPC envelopes return
specific exceptions or request errors. Broad exception handling may contain a
service failure but may not silently convert errors into a normal length finish.

## Documentation and Public Surface

The package exports its supported public Python entrypoints and provides a
documented CLI entrypoint. The default production `LLM` constructor never
silently selects `MockExecutor`; the mock path is explicit and test-only.

`SPEC-ALIGNMENT.md` is replaced by an evidence matrix generated or checked by a
test that verifies every referenced path. Historical claims without current
evidence are removed or clearly marked historical.

## Migration Sequence

1. Add fail-fast request/config validation and progress guarantees.
2. Introduce `BatchPlan` and `ExecutionResult`; migrate EngineCore and runners.
3. Consolidate model loading and executor lifecycle.
4. Move attention construction to the backend registry.
5. Introduce RequestBroker and converge sync/async serving.
6. Replace IPC with persistent engine service and response demultiplexing.
7. Introduce validated ParallelMesh and migrate collectives.
8. Repair package exports, CLI, capability evidence, and architecture docs.

Each migration is test-first and leaves the complete host suite green. Temporary
compatibility adapters may exist within a migration but are removed before phase
one acceptance.

## Test Strategy

Every behavior change follows red-green-refactor. Required host coverage includes:

- oversized-request failure and no-progress detection;
- duplicate request ids and invalid configuration;
- execution contract parity across mock runner modes;
- concurrent IPC streams with request-id isolation;
- cancellation during emission without broker/thread failure;
- bounded router affinity and correct live-load accounting;
- graceful close and executor failure propagation;
- parallel mesh properties and invalid topology rejection;
- extension tests proving model/backend independence;
- a documentation-reference integrity test.

Existing host tests remain green. The initial baseline is 142 passing tests.

## npu2 Validation

Deployment uses a new isolated directory below `/data2`. Before launch, NPU
availability is checked and devices already in use are not selected. Validation
uses existing models under `/data2/models`:

1. Run the complete host suite in the target environment.
2. Run single-card eager and paged correctness on a small available checkpoint.
3. Verify eager/paged/graph token parity where the model supports graph mode.
4. Exercise concurrent serving, cancellation, and process IPC on real NPU.
5. Run real HCCL TP2 and EP2/SP2 topology tests.
6. Run Moonlight-16B MLA/MoE through the persistent engine path.

Commands and logs are retained in the isolated evaluation directory. A failed
test is reported as a failure; it is not reclassified as hardware-gated without
the corresponding device/runtime evidence.

## Phase-one Acceptance

- All original and new host tests pass.
- The oversized-request reproduction terminates with the expected error.
- Concurrent IPC never cross-routes tokens and preserves one persistent engine.
- All service threads/processes close cleanly.
- Every configured parallel axis is represented and validated.
- Eager, paged, graph, and MTP paths use the common execution contract.
- Model and backend extension contracts are enforced by tests.
- Every capability-document reference exists.
- The defined npu2 correctness and stability matrix passes, with retained logs.

## Phase-two Comparison

After phase one, compare auto-infer, omni-npu, and vllm-ascend on the same server,
model, NPU allocation, prompts, request concurrency, batch limits, warmup, and
measurement window. Record TTFT, TPOT, request throughput, token throughput,
peak memory, failures, and variance. Separately compare architectural extension
cost using the files and interfaces changed to add a model, attention backend,
transport, and parallel axis. No framework-quality superiority claim is made
until both runtime and extension evidence are available.
