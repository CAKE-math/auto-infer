# Production Tensor-Parallel Serving Design

## Goal

Add single-node BF16 tensor-parallel serving to the production `main` line
without regressing the existing single-card, FastAPI, continuous-batching,
prefix-cache, or zero-host-bubble paths.

The first production target is Qwen2/Qwen2.5/Qwen3 dense GQA on 2–8 Ascend
NPUs, including Qwen2.5-72B on eight 910B cards. Quantization stays disabled
with an explicit reserved interface. TP speculative decoding remains an
explicitly unsupported capability until its separately trained MTP layer,
KV cache, and graph capture pass a dedicated numerical gate.

## Non-negotiable invariants

1. `tp_size == 1` follows the current construction, scheduling, serving, and
   execution paths without a new thread, process, collective, or branch in a
   decode hot loop.
2. The production HTTP surface remains `ApiRuntime` + FastAPI + `AsyncEngine`;
   TP code must not assign private API-server globals or duplicate protocol
   handling.
3. Every rank owns a rank-local model shard and rank-local KV cache. Ranks see
   the same ordered request-control stream and make identical scheduler/KV
   decisions.
4. A TP replica is one fault domain. A rank may not locally rebuild its
   `EngineCore` after a distributed execution error while peers continue.
5. Steady-state decode adds no Gloo/Python-object collective per step. Control
   synchronization occurs only for submit, abort, and shutdown transitions.
6. BF16 is the only enabled weight format. Model-package quantization remains
   `{enabled: false, interface: reserved}`.
7. Unsupported model, mode, topology, or non-divisible tensor dimensions fail
   before allocating model weights or KV cache.

## Considered approaches

### A. Per-step SPMD object broadcast

Every rank runs a full engine and rank 0 broadcasts submits, aborts, and a seed
with `broadcast_object_list` before every step.

This is easy to reason about, but it creates an unavoidable host barrier in the
decode loop, serializes Python objects continuously, and prevents the current
async pipeline from hiding host scheduling. It is rejected.

### B. Epoch-tagged change-only control plane

Every rank runs the same deterministic `EngineCore`. Rank 0 publishes an
immutable control batch only when requests are submitted, aborted, or the
replica stops. Active replicas apply a batch at a future engine epoch, giving
the receiver thread one complete device step to deliver it before the engine
thread reaches the application boundary. Idle replicas apply immediately.

Follower receiver threads acknowledge delivery, not execution. Rank 0 never
introduces a control batch into its scheduler until all live followers have
acknowledged the same sequence. In a steady decode interval with no request
changes, no host communication is added. This is the selected design.

### C. Rank-0 scheduler with device-side BatchPlan broadcast

Only rank 0 schedules; fixed-layout device buffers containing the next
`BatchPlan` are broadcast to follower executors on a dedicated stream.

This is the long-term design if replicated host scheduling becomes measurable,
but it changes the ownership contract of `EngineCore`, every stager, graph
metadata, and async optimistic state. It is deliberately deferred until the
change-only SPMD implementation has production measurements.

## Architecture

### Model capability and loading

`auto_infer.models.parallel` owns a small model-independent contract:

- `TensorParallelSpec` validates rank, size, head counts, KV-head counts,
  intermediate width, and vocabulary layout.
- A weight-slice callback returns `(dimension, start, length)` for tensors that
  are column- or row-parallel and `None` for replicated tensors.
- The callback operates on checkpoint names before QKV and gate/up packing, so
  packed execution tensors are assembled from already local shards.

`Qwen2Model` exposes the GQA tensor-parallel spec; Qwen3 and MiMo inherit it.
DeepSeek/MLA remains EP-only and is rejected for TP rather than receiving
unknown loader arguments.

`load_sharded` accepts the optional slice callback and reads only that range
from Safetensors. It never materializes the full parallel tensor on host or
NPU. Embeddings and the language-model head remain replicated in the first
correctness milestone. `BaseCausalLM` exposes a `logits_partition` seam so a
later vocab-parallel implementation can replace this without changing
schedulers or Serving.

### Executor composition

The executor registry remains the only execution-mode composition root.
`executor_backends._common()` obtains `tp_rank` and `tp_size` from initialized
parallel state and passes them to executor constructors. `factory.load_model`
resolves the model class, validates its declared parallel capability, and then
passes the accepted TP/EP arguments.

Distributed initialization happens once in `build_executor`, before model
loading, as it does today. TP Serving must not initialize a second overlapping
HCCL world or a second copy of parallel state.

Initial supported mode matrix:

| Model family | recompute | paged | graph | graph_mtp |
|---|---:|---:|---:|---:|
| Qwen dense GQA, TP=1 | yes | yes | yes | existing behavior |
| Qwen dense GQA, TP>1 | yes | yes | yes after NPU graph gate | rejected |
| DeepSeek MLA/MoE, TP>1 | rejected; use EP | rejected | rejected | rejected |

### Change-only control plane

`auto_infer.serving.tp_control` contains transport-neutral immutable messages:

- `ControlBatch(sequence, apply_epoch, submits, aborts, shutdown)`
- `ControlAck(rank, sequence, error)`
- `ReplicaFatal(rank, phase, message)`

The single-node transport uses one multiprocessing queue per follower plus one
ack/status queue. A follower receiver thread consumes and acknowledges messages
independently of device execution. The engine thread applies only the next
contiguous sequence and only when its `apply_epoch` is reached.

Rank 0 keeps the existing bounded submission queue and cancellation accounting.
It converts drained production `EngineService` control into one `ControlBatch`.
Followers do not own HTTP response sinks; they still maintain the same request,
scheduler, output-token, prefix-cache, and KV state.

The service increments `engine_epoch` after each completed `EngineCore.step`.
When work is active, newly drained control is tagged for
`current_epoch + 2`. A follower may already have entered the immediately next
step when rank 0 observes the new request; the extra epoch guarantees that step
uses the old state on every rank and leaves one complete device interval for
delivery. Rank 0 applies the delivered batch at the same later boundary as
followers. When every rank is idle, the batch is applied immediately because no
rank can be inside a model collective.

### Production Serving integration

`AsyncEngine` gains construction from an already-created service while keeping
its current public constructor unchanged. `api_server` separates:

1. engine configuration,
2. `ApiRuntime` construction,
3. Uvicorn execution.

Single-card `serve()` uses these functions exactly as before. TP rank 0 supplies
an `AsyncEngine` backed by the SPMD service to the same runtime builder.
Authentication, admission, async tokenization, SSE, metrics, health, prefix
cache reporting, and graceful shutdown therefore remain one implementation.

Follower ranks run only the SPMD service and control receiver. They never load a
tokenizer or bind an HTTP socket.

### Replica supervision and failure semantics

`auto_infer.serving.tp_server` owns the long-running single-node replica:

- validate `tp_size`, physical device uniqueness, supported mode, and model
  capability before spawning;
- set visibility and HCCL graph environment before importing `torch_npu`;
- start one process per logical rank;
- wait until every rank reports ready before rank 0 reports HTTP readiness;
- on a rank startup/runtime error, publish a fatal status and terminate every
  remaining process;
- treat an unexpected clean rank exit as a replica failure while another rank
  is live;
- bound termination and join time so failed jobs do not retain HBM.

Distributed execution errors bypass `EngineService._recover`; the SPMD service
marks itself unhealthy, fails rank-0 output sinks, reports fatal status, and
exits. The parent supervisor is the only restart boundary.

`HCCL_OP_EXPANSION_MODE=AIV` is enabled for TP graph mode before NPU
initialization. Graph support is considered enabled only after capture and
replay pass on the target CANN/torch-npu stack.

## Accuracy and performance gates

### Host gates

1. Existing test suite remains green.
2. Slice-at-read tensors equal load-full-then-slice tensors for row and column
   partitions, including optional QKV bias and tied embeddings.
3. Invalid rank, duplicate device, unsupported model/mode, and non-divisible
   dimensions fail before loader execution.
4. A two-process host test proves submit, continuous batching, abort, shutdown,
   sequence ordering, and follower state convergence.
5. A follower exception makes the supervisor terminate every child and marks
   rank 0 unhealthy.
6. A steady request with no control changes sends zero additional control
   batches across decode steps.

### NPU gates on `npu2:/data2`

1. Qwen small-model TP2 BF16 greedy output is token-identical to TP1 for
   prefill, decode, continuous batching, prefix-cache hit, and graph replay.
2. Rank-local HBM during loading is bounded by its parallel shard plus
   replicated embedding/head and runtime buffers; no rank materializes the full
   72B checkpoint.
3. Qwen2.5-72B TP8 starts, serves the production HTTP endpoint, and completes
   concurrent greedy requests without rank drift.
4. A killed follower causes the complete replica to exit within the configured
   supervisor timeout and releases all NPU memory.
5. Profiling shows no per-step Gloo object broadcast and no new steady-state
   host gap between graph replays.

## Deferred work with preserved interfaces

- Vocab-parallel LM head and distributed greedy reduction use
  `logits_partition`; the first milestone keeps the head replicated for a
  smaller correctness surface.
- TP recurrent MTP uses the same model capability contract, but remains rejected
  until its attention, FFN reduction, KV layout, and graph capture have an
  independent TP1-versus-TPN token-level gate.
- Quantized TP supplies a future weight-slice transformation behind the same
  loader callback; no quantized execution is enabled now.
- Multi-node TP reuses the control message types but requires a non-local
  transport and topology-aware supervisor, so this design supports one node.
