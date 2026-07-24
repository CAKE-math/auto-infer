from types import SimpleNamespace

import pytest
import torch

from auto_infer.config import SpecDecodeConfig
from auto_infer.spec_decode.geometry import (
    MtpGeometry, validate_graph_mtp_depth)
from auto_infer.worker.graph_mtp_runner import (
    _TargetGear, _decode_packed_results)
from auto_infer.worker.mtp_runner import MtpDrafter, MtpItem


class _FakeHead:
    model = SimpleNamespace(w={
        "lm_head.weight": torch.eye(32, dtype=torch.bfloat16)})

    def forward(self, hidden, token_ids, positions, slots, block_table, cu, kvlens):
        del hidden, positions, slots, block_table, cu, kvlens
        next_ids = (token_ids + 1) % 32
        state = torch.nn.functional.one_hot(
            next_ids, num_classes=32).to(torch.bfloat16)
        logits = state.clone()
        return state, logits


def _drafter():
    return MtpDrafter(
        _FakeHead(), device=torch.device("cpu"), block_size=4)


def test_speculative_depth_is_positive():
    assert SpecDecodeConfig(3).num_speculative_tokens == 3
    with pytest.raises(ValueError, match="> 0"):
        SpecDecodeConfig(0)


def test_single_trained_layer_supports_requested_recurrent_depth():
    weights = {"model.mtp_layers.0.input_proj.weight": object()}
    geometry = MtpGeometry.recurrent_from_weights(weights, 3)
    assert geometry.draft_depth == 3
    assert geometry.proposal_depth == 3
    assert geometry.trained_layer_count == 1
    assert geometry.query_width == 4


def test_graph_mtp_depth_stops_at_npu_verified_boundary():
    validate_graph_mtp_depth(2)
    with pytest.raises(ValueError, match="verified maximum is 2"):
        validate_graph_mtp_depth(3)


def test_eager_drafter_recurs_for_requested_depth():
    item = MtpItem("a", torch.tensor([[1.0], [2.0]]),
                   [4, 5], [0, 1], [3])
    assert _drafter().draft([item], 3) == {"a": (6, 7, 8)}


def test_eager_drafter_uses_stable_greedy_for_bf16_ties():
    head = _FakeHead()
    head.model = SimpleNamespace(w={"lm_head.weight": torch.tensor([
        [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)})
    head.forward = lambda *args: (
        torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.bfloat16))
    drafter = MtpDrafter(
        head, device=torch.device("cpu"), block_size=4)
    item = MtpItem("a", torch.tensor([[1.0, 0.0]]),
                   [4], [0], [3])

    assert drafter.draft([item], 1) == {"a": (1,)}


def test_graph_buffers_and_result_decode_follow_depth():
    geometry = MtpGeometry(3)
    target = _TargetGear(
        2, max_blocks=8, hidden=4, device=torch.device("cpu"),
        dtype=torch.float32, geometry=geometry)
    assert target.tid.shape == (8,)
    assert target.drafts.shape == (2, 3)
    rows = [[10, 11, 12, 13, 2, 20, 21, 22]]
    emitted, drafts, accepted = _decode_packed_results(
        rows, ["a"], geometry)
    assert emitted == {"a": [10, 11, 12]}
    assert drafts == {"a": [20, 21, 22]}
    assert accepted == 2
