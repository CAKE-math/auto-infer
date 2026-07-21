from types import SimpleNamespace

import pytest
import torch

from scripts.verify_ep_dispatch import (
    _install_all_reduce_reference,
    _logit_parity,
    benchmark_summary,
    summarize,
)


def test_ep_report_requires_parity_and_collective_trace():
    rank_results = [
        {"max_abs_error": 0.01, "allclose": True, "token_identity": True,
         "dispatch_calls": 4, "combine_calls": 4,
         "routed_all_reduce_calls": 0},
        {"max_abs_error": 0.02, "allclose": True, "token_identity": True,
         "dispatch_calls": 4, "combine_calls": 4,
         "routed_all_reduce_calls": 0},
    ]

    summary = summarize(rank_results)

    assert summary == {
        "max_abs_error": 0.02,
        "allclose": True,
        "token_identity": True,
        "dispatch_combine_observed": True,
        "routed_all_reduce_observed": False,
        "passed": True,
    }


def test_ep_report_fails_if_any_rank_misses_combine_or_parity():
    rank_results = [
        {"max_abs_error": 0.06, "allclose": False, "token_identity": False,
         "dispatch_calls": 2, "combine_calls": 1,
         "routed_all_reduce_calls": 0},
    ]

    summary = summarize(rank_results)

    assert summary["dispatch_combine_observed"] is False
    assert summary["allclose"] is False
    assert summary["token_identity"] is False
    assert summary["passed"] is False


def test_benchmark_summary_reports_median_speedup():
    summary = benchmark_summary(
        all_to_all_s=[0.004, 0.006, 0.005],
        all_reduce_s=[0.008, 0.012, 0.010],
    )

    assert summary == {
        "all_to_all_median_ms": 5.0,
        "all_reduce_median_ms": 10.0,
        "all_to_all_speedup": 2.0,
    }


def test_reference_backend_counts_routed_all_reduce_calls(monkeypatch):
    moe = SimpleNamespace(
        _fused_w_ep={0: (object(), object())},
        free_originals=False,
        layer_prefix=lambda i: f"layers.{i}.",
        _gate=lambda router, prefix: (
            torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int64)),
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(n_routed=4, routed_scale=1.0),
        dtype=torch.bfloat16,
        w={"layers.0.mlp.gate.weight": torch.zeros(
            4, 2, dtype=torch.bfloat16)},
        moe=moe,
    )
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.ep_rank", lambda: 0)
    monkeypatch.setattr(
        "auto_infer.distributed.parallel_state.ep_size", lambda: 2)
    monkeypatch.setattr("scripts.verify_ep_dispatch._ep_all_reduce", lambda x: x + 1)
    monkeypatch.setattr(
        "scripts.verify_ep_dispatch._all_reduce_expert_reference",
        lambda *args: torch.zeros(1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "auto_infer.layers.mlp.swiglu_mlp",
        lambda *args: torch.zeros(1, 2, dtype=torch.bfloat16))

    calls = _install_all_reduce_reference(model)
    output = model.moe._fused_ep(torch.zeros(1, 2, dtype=torch.bfloat16), 0)

    assert calls == {"calls": 1}
    assert torch.equal(output, torch.ones_like(output))


def test_logit_parity_reports_argmax_and_margin():
    old = torch.tensor([[1.0, 3.0, 2.0]])
    new = torch.tensor([[1.0, 2.5, 2.6]])

    result = _logit_parity(new, old)

    assert result["old_argmax"] == 1
    assert result["new_argmax"] == 2
    assert result["argmax_identity"] is False
    assert result["max_abs_error"] == pytest.approx(0.6)
    assert result["old_top1_margin"] == pytest.approx(1.0)
    assert result["new_top1_margin"] == pytest.approx(0.1)
