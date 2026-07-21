# Omni-aligned BF16 EP Dispatch/Combine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the EP routed-output all-reduce with Ascend's fused BF16 token dispatch/combine protocol while preserving exact routing semantics and graph padding correctness.

**Architecture:** A small communication adapter owns the two torch-npu fused collectives and their metadata. `MoE` composes gate → dispatch → local grouped GEMM → combine → shared expert, while `ForwardContext` carries a fixed-address live-token mask from padded graph gears to every MoE layer.

**Tech Stack:** Python 3.11, PyTorch, torch-npu/CANN, HCCL, pytest, ACL Graph.

## Global Constraints

- Production EP dispatch supports `torch.bfloat16` only.
- BF16 dispatch uses `quant_mode=0` and `scales=None`.
- Quantization metadata remains in the interface, but a quantized policy raises an explicit unsupported-mode error.
- EP size one keeps the existing fused single-rank path.
- EP size greater than one never silently falls back to routed-output all-reduce.
- Local grouped GEMMs consume dispatch-provided expert counts with `group_list_type=1`.
- Layer parity uses `atol=5e-2, rtol=1e-2`; generation under the same numerical regime requires token identity.

---

### Task 1: EP topology and fused communication adapter

**Files:**
- Create: `auto_infer/layers/moe/ep_dispatch.py`
- Modify: `auto_infer/distributed/parallel_state.py`
- Create: `tests/test_ep_dispatch.py`

**Interfaces:**
- Produces: `ExpertParallelTopology(group, rank, world_size, hccl_comm_name)` and `ep_topology()`.
- Produces: `DispatchResult(hidden_states, dynamic_scale, expand_idx, expert_tokens, ep_recv_counts, tp_recv_counts)`.
- Produces: `NpuMoeDispatchCombine(topology, num_experts, dtype, ops=None)`, `.dispatch(x, expert_ids, active_token_mask=None)`, and `.combine(hidden_states, expert_ids, expert_weights, metadata, active_token_mask=None)`.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_bf16_dispatch_forwards_omni_protocol_fields():
    ops = FakeOps()
    adapter = NpuMoeDispatchCombine(
        ExpertParallelTopology(object(), 1, 2, "hccl-ep"), 64,
        torch.bfloat16, ops=ops)
    result = adapter.dispatch(
        torch.zeros(3, 8, dtype=torch.bfloat16),
        torch.tensor([[1, 2], [3, 4], [5, 6]]),
        torch.tensor([True, True, False]))
    assert ops.dispatch_kwargs["quant_mode"] == 0
    assert ops.dispatch_kwargs["scales"] is None
    assert ops.dispatch_kwargs["group_ep"] == "hccl-ep"
    assert result.expert_tokens.dtype == torch.int64

def test_combine_reuses_dispatch_metadata_by_identity():
    ops = FakeOps()
    adapter = NpuMoeDispatchCombine(
        ExpertParallelTopology(object(), 0, 2, "hccl-ep"), 64,
        torch.bfloat16, ops=ops)
    ids = torch.tensor([[1, 2]])
    mask = torch.tensor([True])
    metadata = adapter.dispatch(
        torch.zeros(1, 8, dtype=torch.bfloat16), ids, mask)
    adapter.combine(metadata.hidden_states, ids,
                    torch.ones(1, 2), metadata, mask)
    assert ops.combine_kwargs["assist_info_for_combine"] is metadata.expand_idx
    assert ops.combine_kwargs["ep_send_counts"] is metadata.ep_recv_counts
    assert ops.combine_kwargs["tp_send_counts"] is metadata.tp_recv_counts
    assert ops.combine_kwargs["expert_scales"].dtype == torch.float32

def test_quantized_or_invalid_topology_fails_before_forward():
    with pytest.raises(NotImplementedError, match="BF16"):
        NpuMoeDispatchCombine(topology, 64, torch.int8, ops=FakeOps())
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_ep_dispatch.py`

Expected: collection fails because `auto_infer.layers.moe.ep_dispatch` does not exist.

- [ ] **Step 3: Implement topology caching and the BF16 adapter**

```python
@dataclass(frozen=True)
class ExpertParallelTopology:
    group: object
    rank: int
    world_size: int
    hccl_comm_name: str | None

@dataclass(frozen=True)
class DispatchResult:
    hidden_states: torch.Tensor
    dynamic_scale: torch.Tensor | None
    expand_idx: torch.Tensor
    expert_tokens: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
```

Resolve `_EP_HCCL_COMM_NAME` exactly once after `_EP_GROUP` is selected using
`group._get_backend(torch.device("npu")).get_hccl_comm_name(_EP_RANK)`. Validate
the two v2 operators, BF16 dtype, topology, expert divisibility, result arity,
and mask shape before calling the injected/default torch-npu module.

- [ ] **Step 4: Run adapter and topology tests and verify GREEN**

Run: `pytest -q tests/test_ep_dispatch.py tests/test_parallel_mesh.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the adapter slice**

```bash
git add auto_infer/distributed/parallel_state.py auto_infer/layers/moe/ep_dispatch.py tests/test_ep_dispatch.py
git commit -m "feat: add BF16 EP dispatch combine adapter"
```

### Task 2: Already-dispatched local expert compute

**Files:**
- Modify: `auto_infer/layers/moe/fused_moe.py`
- Create: `tests/test_ep_local_experts.py`

**Interfaces:**
- Consumes: `DispatchResult.hidden_states` and `DispatchResult.expert_tokens`.
- Produces: `fused_local_experts(x, expert_tokens, w13, w2) -> torch.Tensor`.

- [ ] **Step 1: Write a failing local-compute test with fake torch-npu**

```python
def test_local_experts_use_dispatch_counts_without_rerouting(monkeypatch):
    fake = FakeTorchNpu()
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    counts = torch.tensor([2, 3], dtype=torch.int64)
    fused_local_experts(torch.zeros(5, 8), counts, w13, w2)
    assert [call["group_list"] for call in fake.gmm_calls] == [counts, counts]
    assert all(call["group_list_type"] == 1 for call in fake.gmm_calls)
    assert fake.routing_calls == 0
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_ep_local_experts.py`

Expected: import fails because `fused_local_experts` is absent.

- [ ] **Step 3: Implement the two-GMM primitive**

```python
def fused_local_experts(x, expert_tokens, w13, w2):
    import torch_npu
    gate_up = torch_npu.npu_grouped_matmul(
        [x], [w13], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=1,
        output_dtype=x.dtype)[0]
    inter = torch_npu.npu_swiglu(gate_up)
    return torch_npu.npu_grouped_matmul(
        [inter], [w2], bias=None, group_list=expert_tokens,
        split_item=3, group_type=0, group_list_type=1,
        output_dtype=x.dtype)[0]
```

- [ ] **Step 4: Run focused and existing fused-MoE tests**

Run: `pytest -q tests/test_ep_local_experts.py tests/test_deepseek_v2_forward.py`

Expected: all collected tests pass.

- [ ] **Step 5: Commit the compute slice**

```bash
git add auto_infer/layers/moe/fused_moe.py tests/test_ep_local_experts.py
git commit -m "feat: compute dispatched experts locally"
```

### Task 3: Compose true EP in MoE and propagate execution context

**Files:**
- Modify: `auto_infer/forward_context.py`
- Modify: `auto_infer/models/base.py`
- Modify: `auto_infer/models/deepseek_v2.py`
- Modify: `auto_infer/models/qwen2.py`
- Modify: `auto_infer/layers/moe/moe.py`
- Modify: `tests/test_forward_context.py`
- Create: `tests/test_moe_ep_composition.py`

**Interfaces:**
- `ForwardContext.active_token_mask: torch.Tensor | None = None`.
- `_ffn(i, x, prefix, ctx)` forwards context only where the model uses MoE.
- `MoE.__call__(x, layer_idx, active_token_mask=None)`.

- [ ] **Step 1: Write failing context and composition tests**

```python
def test_forward_context_has_optional_active_token_mask():
    assert "active_token_mask" in {
        field.name for field in dataclasses.fields(ForwardContext)}

def test_fused_ep_dispatches_computes_and_combines_without_all_reduce(monkeypatch):
    calls = []
    monkeypatch.setattr(moe_module, "ep_size", lambda: 2)
    monkeypatch.setattr(moe_module, "ep_rank", lambda: 0)
    monkeypatch.setattr(moe_module, "fused_local_experts",
                        lambda x, counts, w13, w2: calls.append("compute") or x)
    block = make_tiny_moe(dispatcher=FakeDispatcher(calls))
    output = block(torch.zeros(2, 8, dtype=torch.bfloat16), 0,
                   torch.tensor([True, False]))
    assert calls == ["dispatch", "compute", "combine"]
    assert output.shape == (2, 8)

def test_ep_rejects_non_divisible_expert_count():
    with pytest.raises(ValueError, match="divisible"):
        NpuMoeDispatchCombine(
            ExpertParallelTopology(object(), 0, 3, "hccl-ep"), 64,
            torch.bfloat16, ops=FakeOps())
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_forward_context.py tests/test_moe_ep_composition.py`

Expected: missing field/signature and old all-reduce composition failures.

- [ ] **Step 3: Replace `_fused_ep` composition and remove production dummy routing**

```python
metadata = self._ep_dispatch.dispatch(x, topk_i, active_token_mask)
local = fused_local_experts(
    metadata.hidden_states, metadata.expert_tokens, w13, w2)
out = self._ep_dispatch.combine(
    local, topk_i, topk_w, metadata, active_token_mask)
return out + swiglu_mlp(x, w, p + "shared_experts.")
```

Keep the old masked-local/all-reduce implementation private and test-only; it
must not be reachable from `_compute`. Cache one adapter per `MoE` instance and
keep local stacked weights per layer.

- [ ] **Step 4: Run model, context, and MoE tests and verify GREEN**

Run: `pytest -q tests/test_forward_context.py tests/test_moe_ep_composition.py tests/test_deepseek_v2_forward.py tests/test_qwen2_forward.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the MoE composition slice**

```bash
git add auto_infer/forward_context.py auto_infer/models/base.py auto_infer/models/deepseek_v2.py auto_infer/models/qwen2.py auto_infer/layers/moe/moe.py tests/test_forward_context.py tests/test_moe_ep_composition.py
git commit -m "feat: route MoE through fused EP collectives"
```

### Task 4: Fixed-address active masks for graph and continuous batching

**Files:**
- Modify: `auto_infer/worker/graph_decode_runner.py`
- Modify: `auto_infer/worker/graph_mtp_gears.py`
- Modify: `auto_infer/worker/graph_mtp_runner.py`
- Modify: `auto_infer/worker/decode_input_stager.py`
- Modify: `auto_infer/worker/prefill_input_stager.py`
- Modify: `tests/test_decode_input_stager.py`
- Modify: `tests/test_graph_decode_runner.py`
- Modify: `tests/test_spec_decode.py`

**Interfaces:**
- Decode and prefill gear objects own token-shaped `active_token_mask` tensors.
- Stagers update masks in-place and return the same fixed-address tensor.
- Spec target masks expand each request mask over `query_width`. The MTP drafter
  and continuation use dense MLPs rather than routed MoE, so they need no EP mask.

- [ ] **Step 1: Write failing mask lifecycle tests**

```python
def test_decode_stager_updates_fixed_address_live_token_mask():
    mask = torch.zeros(4, dtype=torch.bool)
    stager = _stager(active_token_mask=mask)
    ptr = mask.data_ptr()
    stager.stage(_plan(include_b=True))
    assert mask.tolist() == [1, 1, 0, 0]
    stager.stage(_plan(include_b=False))
    assert mask.tolist() == [1, 0, 0, 0]
    assert mask.data_ptr() == ptr

def test_spec_request_mask_expands_to_query_rows():
    mask = torch.zeros(8, dtype=torch.bool)
    stager = make_spec_stager(
        gear=2, geometry=MtpGeometry(3), active_token_mask=mask)
    stager.stage(one_request_plan())
    assert mask.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
```

- [ ] **Step 2: Run graph/stager tests and verify RED**

Run: `pytest -q tests/test_decode_input_stager.py tests/test_graph_decode_runner.py tests/test_spec_decode.py`

Expected: constructors or mask assertions fail because token masks are not wired.

- [ ] **Step 3: Add mask buffers and pass them through every graph context**

Use `torch.bool` fixed-address device buffers, matching omni-npu's MC2 contract.
Decode masks mark the first live
requests; prefill masks mark the first `real_query_tokens`; speculative target
masks repeat each request bit by `query_width`. The speculative gear retains its
existing request-level `int32 active_mask` for acceptance arithmetic and owns a
separate token-level bool `ep_active_token_mask`. Eager contexts keep `None`.
Reject a padded graph context whose mask is absent, non-bool, or has a different
number of elements from `token_ids`.

- [ ] **Step 4: Run graph/stager tests and verify GREEN**

Run: `pytest -q tests/test_decode_input_stager.py tests/test_prefill_input_stager.py tests/test_graph_decode_runner.py tests/test_spec_decode.py`

Expected: all tests pass and mask data pointers remain stable across stages.

- [ ] **Step 5: Commit the graph mask slice**

```bash
git add auto_infer/worker/graph_decode_runner.py auto_infer/worker/graph_mtp_gears.py auto_infer/worker/graph_mtp_runner.py auto_infer/worker/decode_input_stager.py auto_infer/worker/prefill_input_stager.py tests/test_decode_input_stager.py tests/test_graph_decode_runner.py tests/test_spec_decode.py
git commit -m "feat: mask padded EP graph tokens"
```

### Task 5: Regression, npu2 parity, and performance evidence

**Files:**
- Create: `scripts/verify_ep_dispatch.py`
- Create: `tests/test_verify_ep_dispatch.py`
- Create: `docs/PERFORMANCE-EP-DISPATCH.md`

**Interfaces:**
- The script accepts model path, EP size, prompt/generation lengths, warmups,
  repeats, graph/eager mode, and writes a JSON result per rank plus a summary.

- [ ] **Step 1: Add a host test for CLI parsing and result aggregation**

```python
def test_ep_report_requires_parity_and_collective_trace():
    rank_results = [
        {"max_abs_error": 0.01, "token_identity": True,
         "dispatch_calls": 4, "combine_calls": 4,
         "routed_all_reduce_calls": 0},
        {"max_abs_error": 0.02, "token_identity": True,
         "dispatch_calls": 4, "combine_calls": 4,
         "routed_all_reduce_calls": 0},
    ]
    summary = summarize(rank_results)
    assert summary["max_abs_error"] <= 5e-2
    assert summary["token_identity"] is True
    assert summary["dispatch_combine_observed"] is True
    assert summary["routed_all_reduce_observed"] is False
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest -q tests/test_verify_ep_dispatch.py`

Expected: import fails because the verification module is absent.

- [ ] **Step 3: Implement the verifier and concise report template**

The verifier runs the test-only all-reduce reference and fused dispatch path on
identical BF16 inputs, checks layer tolerance, checks Moonlight token identity,
records throughput/step communication time, and emits exact command/model
revision/topology fields. The report states measured values only.

- [ ] **Step 4: Run the complete host suite**

Run: `pytest -q`

Expected: zero failures.

- [ ] **Step 5: Run npu2 EP2 and EP4 acceptance**

Deploy into a new isolated directory and run inside the recorded Ascend
container:

```bash
ssh npu2 'mkdir -p /data2/auto-infer-ep-dispatch-20260721'
rsync -az --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' ./ npu2:/data2/auto-infer-ep-dispatch-20260721/
ssh npu2 'docker exec -w /data2/auto-infer-ep-dispatch-20260721 auto-infer-dev-20260624 torchrun --nproc-per-node=2 scripts/verify_ep_dispatch.py --model /data2/models/Moonlight-16B-A3B-Instruct --ep-size 2 --dtype bfloat16 --mode eager'
ssh npu2 'docker exec -w /data2/auto-infer-ep-dispatch-20260721 auto-infer-dev-20260624 torchrun --nproc-per-node=2 scripts/verify_ep_dispatch.py --model /data2/models/Moonlight-16B-A3B-Instruct --ep-size 2 --dtype bfloat16 --mode graph'
ssh npu2 'docker exec -w /data2/auto-infer-ep-dispatch-20260721 auto-infer-dev-20260624 torchrun --nproc-per-node=4 scripts/verify_ep_dispatch.py --model /data2/models/Moonlight-16B-A3B-Instruct --ep-size 4 --dtype bfloat16 --mode graph'
```

Expected: layer parity within `atol=5e-2, rtol=1e-2`, token identity, dispatch
and combine v2 observed, routed-output all-reduce absent, and all ranks exit 0.

- [ ] **Step 6: Record results and commit verification evidence**

```bash
git add scripts/verify_ep_dispatch.py tests/test_verify_ep_dispatch.py docs/PERFORMANCE-EP-DISPATCH.md
git commit -m "test: validate fused EP dispatch on Ascend"
```

### Task 6: Final architecture and cleanliness gate

**Files:**
- Modify only files identified by the checks below.

- [ ] **Step 1: Verify no production all-reduce remains in fused EP composition**

Run: `rg -n "ep_all_reduce|fused_experts_ep" auto_infer/layers/moe`

Expected: references occur only in the explicitly named private test reference;
`MoE._fused_ep` contains neither symbol.

- [ ] **Step 2: Run static and full regression checks**

Run: `python -m compileall -q auto_infer scripts tests && git diff --check && pytest -q`

Expected: exit 0 and zero pytest failures.

- [ ] **Step 3: Review final diff against every design requirement**

Run: `git diff a5e6bce --stat && git status --short`

Expected: only EP adapter, local compute, context/mask propagation, tests,
verification script, design/plan, and performance evidence are present.

- [ ] **Step 4: Squash the implementation history for the repository's single-commit policy**

Create a replacement root commit containing the verified tree, preserve `main`
as the only branch, and force-push only after comparing the replacement tree ID
with the verified pre-squash tree ID.
