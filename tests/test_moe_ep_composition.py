from types import SimpleNamespace

import torch

from auto_infer.distributed import parallel_state as ps
from auto_infer.layers.moe import fused_moe
from auto_infer.layers.moe.ep_dispatch import DispatchResult
from auto_infer.layers.moe.moe import MoE


def _config(n_routed=4):
    return SimpleNamespace(
        n_routed=n_routed,
        top_k=2,
        scoring_func="softmax",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scale=1.0,
    )


def _weights(n_routed=4, hidden=4, intermediate=3):
    weights = {
        "model.layers.0.mlp.gate.weight": torch.arange(
            n_routed * hidden, dtype=torch.bfloat16).view(n_routed, hidden),
    }
    for expert in range(n_routed):
        prefix = f"model.layers.0.mlp.experts.{expert}."
        weights[prefix + "gate_proj.weight"] = torch.zeros(
            intermediate, hidden, dtype=torch.bfloat16)
        weights[prefix + "up_proj.weight"] = torch.zeros(
            intermediate, hidden, dtype=torch.bfloat16)
        weights[prefix + "down_proj.weight"] = torch.zeros(
            hidden, intermediate, dtype=torch.bfloat16)
    for name, shape in (
        ("gate_proj.weight", (intermediate, hidden)),
        ("up_proj.weight", (intermediate, hidden)),
        ("down_proj.weight", (hidden, intermediate)),
    ):
        weights["model.layers.0.mlp.shared_experts." + name] = torch.zeros(
            *shape, dtype=torch.bfloat16)
    return weights


class _FakeDispatcher:
    def __init__(self, calls):
        self.calls = calls
        self.dispatch_mask = None
        self.combine_mask = None

    def dispatch(self, x, expert_ids, active_token_mask):
        self.calls.append("dispatch")
        self.dispatch_mask = active_token_mask
        assert expert_ids.dtype == torch.int32
        return DispatchResult(
            hidden_states=x,
            dynamic_scale=None,
            expand_idx=torch.tensor([0], dtype=torch.int32),
            expert_tokens=torch.tensor([x.shape[0], 0], dtype=torch.int64),
            ep_recv_counts=torch.tensor([1, 0], dtype=torch.int32),
            tp_recv_counts=torch.tensor([0, 0], dtype=torch.int32),
        )

    def combine(self, hidden, expert_ids, expert_weights, metadata,
                active_token_mask):
        self.calls.append("combine")
        self.combine_mask = active_token_mask
        assert expert_ids.dtype == torch.int32
        assert expert_weights.dtype == torch.bfloat16
        return torch.full_like(hidden, 3)


def test_fused_ep_dispatches_computes_and_combines_without_all_reduce(monkeypatch):
    calls = []
    dispatcher = _FakeDispatcher(calls)
    block = MoE(
        _weights(), _config(), torch.device("cpu"), torch.bfloat16,
        free_originals=False)
    block._ep_dispatch = dispatcher
    mask = torch.tensor([True, False])

    monkeypatch.setattr(ps, "ep_size", lambda: 2)
    monkeypatch.setattr(ps, "ep_rank", lambda: 0)
    assert not hasattr(ps, "ep_all_reduce")

    def local_compute(x, counts, w13, w2):
        calls.append("compute")
        assert counts.tolist() == [2, 0]
        assert w13.shape[0] == w2.shape[0] == 2
        return x

    monkeypatch.setattr(fused_moe, "fused_local_experts", local_compute,
                        raising=False)
    monkeypatch.setattr(
        "auto_infer.layers.moe.moe.swiglu_mlp",
        lambda x, weights, prefix: torch.full_like(x, 5))

    output = block(
        torch.ones(2, 4, dtype=torch.bfloat16), 0,
        active_token_mask=mask)

    assert calls == ["dispatch", "compute", "combine"]
    assert dispatcher.dispatch_mask is mask
    assert dispatcher.combine_mask is mask
    assert torch.equal(output, torch.full_like(output, 8))


def test_ep_adapter_is_created_once_per_moe_instance(monkeypatch):
    created = []
    dispatcher = _FakeDispatcher([])
    block = MoE(
        _weights(), _config(), torch.device("cpu"), torch.bfloat16,
        free_originals=False)
    monkeypatch.setattr(ps, "ep_size", lambda: 2)
    monkeypatch.setattr(ps, "ep_rank", lambda: 0)
    monkeypatch.setattr(ps, "ep_topology", lambda: object(), raising=False)
    monkeypatch.setattr(
        "auto_infer.layers.moe.ep_dispatch.NpuMoeDispatchCombine",
        lambda topology, num_experts, dtype: created.append(
            (topology, num_experts, dtype)) or dispatcher)
    monkeypatch.setattr(
        fused_moe, "fused_local_experts", lambda x, counts, w13, w2: x,
        raising=False)
    monkeypatch.setattr(
        "auto_infer.layers.moe.moe.swiglu_mlp",
        lambda x, weights, prefix: torch.zeros_like(x))

    x = torch.ones(2, 4, dtype=torch.bfloat16)
    block(x, 0)
    block(x, 0)

    assert len(created) == 1
    assert created[0][1:] == (4, torch.bfloat16)
