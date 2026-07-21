import sys

import torch

from auto_infer.layers.moe.fused_moe import fused_local_experts


class _FakeTorchNpu:
    def __init__(self):
        self.gmm_calls = []
        self.swiglu_calls = 0

    def npu_grouped_matmul(self, xs, weights, **kwargs):
        self.gmm_calls.append({"xs": xs, "weights": weights, **kwargs})
        rows = xs[0].shape[0]
        width = weights[0].shape[-1]
        return [torch.zeros(rows, width, dtype=xs[0].dtype)]

    def npu_swiglu(self, x):
        self.swiglu_calls += 1
        return x[:, :x.shape[-1] // 2]


def test_local_experts_use_dispatch_counts_without_rerouting(monkeypatch):
    fake = _FakeTorchNpu()
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    counts = torch.tensor([2, 3], dtype=torch.int64)
    x = torch.zeros(5, 8, dtype=torch.bfloat16)
    w13 = torch.zeros(2, 8, 12, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 6, 8, dtype=torch.bfloat16)

    output = fused_local_experts(x, counts, w13, w2)

    assert output.shape == (5, 8)
    assert len(fake.gmm_calls) == 2
    assert [call["group_list"] for call in fake.gmm_calls] == [counts, counts]
    assert all(call["group_list_type"] == 1 for call in fake.gmm_calls)
    assert all(call["group_type"] == 0 for call in fake.gmm_calls)
    assert all(call["split_item"] == 3 for call in fake.gmm_calls)
    assert all(call["output_dtype"] == torch.bfloat16
               for call in fake.gmm_calls)
    assert fake.swiglu_calls == 1
    assert not hasattr(fake, "npu_moe_init_routing")
    assert not hasattr(fake, "npu_moe_finalize_routing")


def test_local_experts_reject_count_or_weight_contract_mismatch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", _FakeTorchNpu())
    x = torch.zeros(5, 8, dtype=torch.bfloat16)
    w13 = torch.zeros(2, 8, 12, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 6, 8, dtype=torch.bfloat16)

    for counts in (
        torch.tensor([2, 3], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int64),
    ):
        try:
            fused_local_experts(x, counts, w13, w2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid expert count contract was accepted")
