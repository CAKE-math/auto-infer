# auto-infer

Independent Ascend-NPU LLM inference runtime focused on a small core, explicit
execution contracts, paged KV caching, and model-independent attention backends.

The current acceptance target is architecture convergence, correctness, and
stability. Performance comparisons with omni-npu and vllm-ascend are recorded
only after the same workloads pass that gate.

## Performance

On one Ascend 910B1, the matched final measurements report 2,259.2 tok/s for
Qwen3-0.6B at batch size 16, 228.99 tok/s for Moonlight-16B-A3B at batch size 4,
and 895.51 tok/s for MiMo-7B MTP at batch size 16. Workload definitions,
latency, stability, correctness caveats, and competitor results are in the
[concise performance report](docs/PERFORMANCE-REPORT-2026-07-20.md).

## Quick start

Install the Python package without replacing the vendor-provided `torch` and
`torch_npu`, then run the host suite:

```bash
pip install -e .
pytest -q
auto-infer --help
```

Start the OpenAI-compatible server on an Ascend host:

```bash
auto-infer serve /path/to/model --device 0 --mode paged --port 8000
```

Supported execution modes are `recompute`, `paged`, `graph`, and `graph_mtp`.
All capacity, dtype, device, and scheduling settings flow through one
`EngineConfig`; test code can opt into `LLM.for_testing()` explicitly.

## Architecture

```text
LLM / HTTP / IPC
       │
EngineService ─ RequestBroker
       │
EngineCore ─ Scheduler ─ KVCacheManager
       │ BatchPlan             ▲
       ▼                       │ ExecutionResult
Executor factory ──────────────┘
  ├─ recompute
  ├─ paged FIA
  ├─ ACL-graph decode
  └─ graph MTP
       │
BaseCausalLM + attention registry (GQA / MLA)
```

The engine owns mutable request and KV state. Executors receive immutable
`BatchPlan` snapshots and return `ExecutionResult`; runners do not reach back
into the scheduler. One persistent `EngineService` owns each engine, while the
broker and IPC demultiplexer handle concurrency and lifecycle. Replica routing
belongs to the deployment layer, not this single-engine runtime.

Parallel groups are derived from an explicit `ParallelMesh` with named TP, DP,
EP, CP, and SP axes. Model subclasses declare an attention family; eager, paged,
and graph backend selection stays in the central attention registry.

## Supported surface

- Qwen2/Qwen3 GQA and DeepSeek-V2/V3-style MLA/MoE model paths.
- Continuous batching, chunked prefill, prefix caching, KV preemption, and
  synchronous or bounded asynchronous scheduling.
- Paged FIA execution, ACL-graph decode, and trained-head MTP execution paths.
- OpenAI-compatible completions/chat endpoints and SSE streaming.
- Persistent process IPC with request-id demultiplexing.
- Explicit TP/DP/EP/CP/SP mesh construction.

These are implementation claims, not universal hardware support claims. Exact
validated combinations and retained evidence are listed in
[the capability matrix](docs/SPEC-ALIGNMENT.md). The current npu2 acceptance and
baseline results are in the [phase-one report](docs/PHASE1-VALIDATION-2026-07-19.md)
and [three-framework comparison](docs/ARCHITECTURE-COMPARISON.md).

## Extension points

Adding a model that reuses GQA or MLA requires a `BaseCausalLM` subclass and a
model-registry entry. A new attention shape adds one backend-family registration;
it does not add factory methods to every model or branches to every runner.
Recurrent MTP attention is a separately registered capability: GQA supports it,
while MLA MTP currently fails explicitly as unsupported. `auto_infer.pd`
retains low-level experimental KV-copy/HCCL operations but is not wired into
the serving lifecycle.

Relevant contracts:

- [engine/execution.py](auto_infer/engine/execution.py)
- [engine/factory.py](auto_infer/engine/factory.py)
- [layers/attention/registry.py](auto_infer/layers/attention/registry.py)
- [distributed/mesh.py](auto_infer/distributed/mesh.py)
- [serving/service.py](auto_infer/serving/service.py)

## Verification

Host verification:

```bash
pytest -q
python -m compileall -q auto_infer
git diff --check
```

NPU verification scripts live under [scripts](scripts/). Environment notes are
in [docs/NPU-ENV.md](docs/NPU-ENV.md). The current architecture rationale and
implementation plan are in the [design](docs/superpowers/specs/2026-07-19-architecture-convergence-design.md)
and [plan](docs/superpowers/plans/2026-07-19-architecture-convergence.md).
