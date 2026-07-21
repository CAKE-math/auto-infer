import pytest
import torch

from auto_infer.distributed import parallel_state as ps
from auto_infer.distributed.topology import ExpertParallelTopology
from auto_infer.layers.moe.ep_dispatch import (
    DispatchResult,
    MoeDispatchQuantization,
    NpuMoeDispatchCombine,
)


class _FakeOps:
    def __init__(self):
        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.expand_idx = torch.tensor([11], dtype=torch.int32)
        self.expert_tokens = torch.tensor([1, 2], dtype=torch.int32)
        self.ep_counts = torch.tensor([3, 4], dtype=torch.int32)
        self.tp_counts = torch.tensor([5, 6], dtype=torch.int32)

    def npu_moe_distribute_dispatch_v2(self, **kwargs):
        self.dispatch_kwargs = kwargs
        return (
            kwargs["x"] + 1,
            None,
            self.expand_idx,
            self.expert_tokens,
            self.ep_counts,
            self.tp_counts,
        )

    def npu_moe_distribute_combine_v2(self, **kwargs):
        self.combine_kwargs = kwargs
        return torch.full(
            (kwargs["expert_ids"].shape[0], kwargs["expand_x"].shape[1]),
            2, dtype=kwargs["expand_x"].dtype)


def _topology(world_size=2, rank=1, name="hccl-ep"):
    return ExpertParallelTopology(object(), rank, world_size, name)


def test_bf16_dispatch_forwards_omni_protocol_fields():
    ops = _FakeOps()
    adapter = NpuMoeDispatchCombine(
        _topology(), num_experts=64, dtype=torch.bfloat16, ops=ops)
    x = torch.zeros(3, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    mask = torch.tensor([True, True, False])

    result = adapter.dispatch(x, ids, mask)

    assert ops.dispatch_kwargs == {
        "x": x,
        "expert_ids": ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": 64,
        "global_bs": 0,
        "scales": None,
        "quant_mode": 0,
        "group_ep": "hccl-ep",
        "ep_world_size": 2,
        "ep_rank_id": 1,
        "x_active_mask": mask,
    }
    assert isinstance(result, DispatchResult)
    assert result.expert_tokens.dtype == torch.int64
    assert result.expand_idx is ops.expand_idx
    assert result.ep_recv_counts is ops.ep_counts
    assert result.tp_recv_counts is ops.tp_counts
    assert result.dynamic_scale is None
    assert adapter.dispatch_calls == 1
    assert adapter.combine_calls == 0


def test_combine_reuses_dispatch_metadata_and_converts_weights_to_fp32():
    ops = _FakeOps()
    adapter = NpuMoeDispatchCombine(
        _topology(rank=0), num_experts=64, dtype=torch.bfloat16, ops=ops)
    ids = torch.tensor([[1, 2]], dtype=torch.int32)
    mask = torch.tensor([True])
    metadata = adapter.dispatch(
        torch.zeros(1, 8, dtype=torch.bfloat16), ids, mask)
    local = torch.zeros(3, 8, dtype=torch.bfloat16)

    output = adapter.combine(
        local, ids, torch.ones(1, 2, dtype=torch.bfloat16), metadata, mask)

    kwargs = ops.combine_kwargs
    assert kwargs["expand_x"] is local
    assert kwargs["expert_ids"] is ids
    assert kwargs["assist_info_for_combine"] is metadata.expand_idx
    assert kwargs["ep_send_counts"] is metadata.ep_recv_counts
    assert kwargs["tp_send_counts"] is metadata.tp_recv_counts
    assert kwargs["expert_scales"].dtype == torch.float32
    assert kwargs["x_active_mask"] is mask
    assert kwargs["group_ep"] == "hccl-ep"
    assert kwargs["ep_world_size"] == 2
    assert kwargs["ep_rank_id"] == 0
    assert output.shape == (1, 8)
    assert adapter.dispatch_calls == 1
    assert adapter.combine_calls == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_non_bf16_policy_is_rejected(dtype):
    with pytest.raises(NotImplementedError, match="BF16"):
        NpuMoeDispatchCombine(_topology(), 64, dtype, ops=_FakeOps())


def test_quantization_policy_is_explicit_but_not_enabled():
    with pytest.raises(NotImplementedError, match="quantized"):
        NpuMoeDispatchCombine(
            _topology(), 64, torch.bfloat16, ops=_FakeOps(),
            quantization=MoeDispatchQuantization(quant_mode=2, scales=None))


@pytest.mark.parametrize(
    ("topology", "message"),
    [
        (_topology(world_size=3), "divisible"),
        (_topology(rank=2, world_size=2), "rank"),
        (ExpertParallelTopology(None, 0, 2, "hccl-ep"), "process group"),
        (_topology(name=None), "communicator"),
    ],
)
def test_invalid_topology_is_rejected(topology, message):
    with pytest.raises(ValueError, match=message):
        NpuMoeDispatchCombine(topology, 64, torch.bfloat16, ops=_FakeOps())


def test_missing_fused_operator_is_rejected_during_construction():
    with pytest.raises(RuntimeError, match="dispatch_v2"):
        NpuMoeDispatchCombine(_topology(), 64, torch.bfloat16, ops=object())


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([True]),
        torch.tensor([[True, True]]),
    ],
)
def test_dispatch_requires_token_shaped_bool_mask(mask):
    adapter = NpuMoeDispatchCombine(
        _topology(), 64, torch.bfloat16, ops=_FakeOps())
    with pytest.raises(ValueError, match="active-token mask"):
        adapter.dispatch(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.tensor([[1], [2]], dtype=torch.int32),
            mask,
        )


def test_dispatch_requires_int32_global_expert_ids():
    adapter = NpuMoeDispatchCombine(
        _topology(), 64, torch.bfloat16, ops=_FakeOps())
    with pytest.raises(ValueError, match="int32"):
        adapter.dispatch(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.tensor([[1], [2]], dtype=torch.int64),
        )


def test_dispatch_rejects_invalid_operator_output_contract():
    ops = _FakeOps()
    ops.npu_moe_distribute_dispatch_v2 = lambda **_: (
        torch.zeros(2, 8, dtype=torch.float16), None,
        ops.expand_idx, ops.expert_tokens, ops.ep_counts, ops.tp_counts)
    adapter = NpuMoeDispatchCombine(
        _topology(), 64, torch.bfloat16, ops=ops)

    with pytest.raises(RuntimeError, match="BF16 matrix"):
        adapter.dispatch(
            torch.zeros(2, 8, dtype=torch.bfloat16),
            torch.tensor([[1], [2]], dtype=torch.int32),
        )


def test_combine_rejects_invalid_operator_output_contract():
    ops = _FakeOps()
    ops.npu_moe_distribute_combine_v2 = lambda **kwargs: torch.zeros(
        kwargs["expert_ids"].shape[0] + 1,
        kwargs["expand_x"].shape[1], dtype=torch.bfloat16)
    adapter = NpuMoeDispatchCombine(
        _topology(), 64, torch.bfloat16, ops=ops)
    ids = torch.tensor([[1, 2]], dtype=torch.int32)
    metadata = adapter.dispatch(
        torch.zeros(1, 8, dtype=torch.bfloat16), ids)

    with pytest.raises(RuntimeError, match="source-token shape"):
        adapter.combine(
            torch.zeros(3, 8, dtype=torch.bfloat16), ids,
            torch.ones(1, 2, dtype=torch.bfloat16), metadata)


def test_parallel_state_exposes_cached_ep_topology(monkeypatch):
    group = object()
    monkeypatch.setattr(ps, "_EP_GROUP", group)
    monkeypatch.setattr(ps, "_EP_RANK", 1)
    monkeypatch.setattr(ps, "_EP_SIZE", 4)
    monkeypatch.setattr(ps, "_EP_HCCL_COMM_NAME", "cached-name", raising=False)

    topology = ps.ep_topology()

    assert topology.group is group
    assert topology.rank == 1
    assert topology.world_size == 4
    assert topology.hccl_comm_name == "cached-name"
