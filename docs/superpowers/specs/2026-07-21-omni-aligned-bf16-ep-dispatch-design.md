# Omni-aligned BF16 expert-parallel dispatch/combine

## Goal

Replace auto-infer's masked-local-expert plus full-output all-reduce path with
the same fused Ascend token dispatch/combine protocol used by omni-npu. The
production path covers BF16 only. Quantization is represented by stable
protocol fields but is not accepted as a supported execution mode.

## Current problem

For EP size greater than one, every rank currently receives the full token
batch, computes only routes owned by its local expert range, produces a
full-shaped mostly-zero output, and all-reduces that output. Communication and
intermediate memory therefore scale with the complete token tensor instead of
the tokens routed to each rank. This is expert weight sharding, but not token
dispatch.

## Production data flow

For one MoE layer and an input `x` of shape `(tokens, hidden)`:

1. The replicated gate computes router logits and the existing DeepSeek
   softmax/sigmoid/group-limited top-k policy produces global expert IDs and
   router weights.
2. `npu_moe_distribute_dispatch_v2` sends each expanded top-k token only to the
   EP rank that owns its expert. It receives the EP HCCL communicator name,
   global expert count, EP rank/size, and the optional active-token mask.
3. Dispatch returns expert-sorted local activations, local expert token counts,
   combine indices, and EP/TP receive counts.
4. Two local grouped GEMMs consume the dispatch-sorted activations directly:
   fused gate/up projection, SwiGLU, then down projection. The group list is the
   dispatch-provided local expert token count tensor and uses
   `group_list_type=1`; routing is not repeated.
5. `npu_moe_distribute_combine_v2` reverse-dispatches local outputs, restores
   source-token order with the combine indices, multiplies FP32 router weights,
   and sums top-k contributions.
6. The replicated shared-expert result is added after routed combine, matching
   the current numerical contract.

The EP path contains no routed-output all-reduce. `ep_size == 1` retains the
existing single-rank fused routing path.

## Components and boundaries

### `auto_infer/layers/moe/ep_dispatch.py`

Owns the fused communication protocol and no model weights.

- `DispatchResult` contains `hidden_states`, `expert_tokens`, `expand_idx`,
  `ep_recv_counts`, `tp_recv_counts`, and optional `dynamic_scale`.
- `NpuMoeDispatchCombine.dispatch(...)` calls dispatch v2 and validates its
  result shape/dtype contract.
- `NpuMoeDispatchCombine.combine(...)` calls combine v2 with the exact metadata
  returned by dispatch.
- The constructor consumes an injected EP topology object so host tests can
  validate calls without initializing HCCL.

BF16 dispatch always uses `quant_mode=0` and `scales=None`. `dynamic_scale`
remains in `DispatchResult` so a later quantization policy can consume it
without changing callers. Passing a non-BF16 quantization policy raises an
explicit unsupported-mode error.

### `auto_infer/distributed/parallel_state.py`

The runtime EP topology exposes rank, world size, process group, and its HCCL
communicator name. The communicator name is resolved once after group creation
and cached; no backend introspection occurs in a MoE layer's forward call.

### `auto_infer/layers/moe/fused_moe.py`

Adds a focused local-expert primitive that consumes already-dispatched,
expert-sorted tokens and `expert_tokens`. It performs only grouped GEMM,
activation, and grouped GEMM. It does not route, finalize, communicate, or
apply router weights.

### `auto_infer/layers/moe/moe.py`

Owns composition: gate, dispatch, local expert compute, combine, and shared
expert addition. It lazily constructs one dispatcher per EP topology and
caches stacked local expert weights per layer. The old EP all-reduce path is
retained only as a private numerical reference used by tests, not as a runtime
fallback.

### Active-token masking

`ForwardContext` gains optional `active_token_mask`. Runners that pad graph
gears provide a fixed-address device mask; eager paths leave it `None` because
all rows are live. The model forwards this mask only to MoE. Dispatch and
combine receive the identical mask so padding rows generate neither network
traffic nor routed output.

## Failure behavior

Invalid topology, unavailable fused operators, and unsupported quantization
fail during executor/model initialization, before the first EP forward:

- the global expert count is not divisible by EP size;
- the EP process group or HCCL communicator name is unavailable;
- dispatch/combine v2 is missing from the installed torch-npu;
- a quantized EP policy is requested.

A padded graph gear without an active-token mask fails while graph inputs are
staged or captured, before that graph is admitted for serving.

There is no silent fallback to full-output all-reduce because that would make
performance and memory behavior configuration-dependent and would conceal a
production topology error.

## Graph and continuous-batching contract

Dispatch/combine tensors and active masks are fixed-address inputs under ACL
graph capture. Dynamic route counts remain device tensors returned by dispatch;
the host does not call `.tolist()`, create split lists, or synchronize on token
counts. Continuous batching may change the number and expert distribution of
live rows while the captured gear shape stays fixed; `active_token_mask`
identifies live rows.

## Testing and acceptance

Host tests with injected torch-npu and topology fakes must prove:

- BF16 dispatch uses `quant_mode=0`, `scales=None`, global expert IDs, cached
  HCCL name, correct EP rank/size, and the exact active mask;
- combine reuses dispatch's `expand_idx`, EP counts, and TP counts by identity,
  converts router weights to FP32, and forwards the same mask;
- local grouped expert compute uses dispatch's expert token counts and performs
  no routing/finalize operation;
- MoE EP composition contains no `ep_all_reduce` call;
- invalid topology, unavailable fused ops, and quantized policies fail early.

On npu2, BF16 EP2 and EP4 tests must compare against the existing all-reduce
reference, first layer-by-layer with `atol=5e-2, rtol=1e-2`, and then through
Moonlight generation with token identity under the same graph/eager numerical
regime. Profiling must show dispatch/combine v2 and no routed-output all-reduce.
The report records throughput, per-step communication time, output parity,
topology, model revision, and exact commands.

## Non-goals

- W8A8/W4A8/HiFloat8 expert execution;
- EPLB or expert replication;
- shared-expert rank sharding;
- a production standard-collective fallback;
- changing router selection mathematics or expert ownership.
