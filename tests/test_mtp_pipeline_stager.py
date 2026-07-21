from types import SimpleNamespace
import inspect

import pytest
import torch

from auto_infer.spec_decode.geometry import MtpGeometry
from auto_infer.worker.mtp_pipeline_stager import MtpPipelineStager
import auto_infer.worker.mtp_pipeline_stager as mtp_pipeline_stager


class _Plan:
    def __init__(self, requests, block_tables):
        self.requests = requests
        self.block_tables = block_tables

    def get_request(self, request_id):
        return self.requests[request_id]


def _request(computed):
    return SimpleNamespace(num_computed_tokens=computed)


def _plan():
    return _Plan(
        {rid: _request(base) for rid, base in zip("abcd", (10, 20, 30, 40))},
        {rid: (index + 1,) for index, rid in enumerate("abcd")})


def _stager():
    return MtpPipelineStager(
        block_table=torch.zeros(5, 8, dtype=torch.int32),
        sample_rows=torch.zeros(4, dtype=torch.long),
        block_size=4, scratch0=100, token_capacity=8,
        geometry=MtpGeometry(1))


def test_mtp_stager_builds_mixed_confirmed_metadata_with_dummy_padding():
    staged = _stager().stage_drafter(
        _plan(), list("abcd"), [0, 1, 0, 1],
        token_gear=8, request_gear=4)

    assert staged.cumulative_query_lengths == [1, 3, 4, 6, 8]
    assert staged.kv_lengths == [11, 22, 31, 42, 2]
    assert staged.sample_rows[:4].tolist() == [0, 2, 3, 5]
    assert staged.block_table.shape == (5, 8)
    assert staged.block_table[:4, 0].tolist() == [1, 2, 3, 4]
    assert staged.block_table[4, 0].item() == 100
    assert staged.padding_blocks == (100,)
    assert staged.request_count == 4
    assert staged.active_tokens == 6


def test_mtp_stager_omits_dummy_for_exact_all_accept_gear():
    staged = _stager().stage_drafter(
        _plan(), list("abcd"), [1, 1, 1, 1],
        token_gear=8, request_gear=4)

    assert staged.cumulative_query_lengths == [2, 4, 6, 8]
    assert staged.kv_lengths == [12, 22, 32, 42]
    assert staged.sample_rows[:4].tolist() == [1, 3, 5, 7]
    assert staged.block_table.shape[0] == 4


def test_mtp_stager_all_reject_padding_is_scratch_only():
    staged = _stager().stage_drafter(
        _plan(), list("abcd"), [0, 0, 0, 0],
        token_gear=8, request_gear=4)

    assert staged.cumulative_query_lengths == [1, 2, 3, 4, 8]
    assert staged.kv_lengths == [11, 21, 31, 41, 4]
    assert staged.block_table[-1, 0].item() == 100
    assert all(block >= 100 for block in staged.padding_blocks)


def test_mtp_stager_reuses_clean_block_table_rows():
    stager = _stager()
    first = stager.stage_drafter(
        _plan(), list("abcd"), [0, 1, 0, 1],
        token_gear=8, request_gear=4)
    copied = stager.copied_block_rows
    second = stager.stage_drafter(
        _plan(), list("abcd"), [0, 1, 0, 1],
        token_gear=8, request_gear=4)

    assert first.block_table.data_ptr() == second.block_table.data_ptr()
    assert stager.copied_block_rows == copied


def test_mtp_stager_does_not_depend_on_graph_runner():
    assert "graph_mtp_runner" not in inspect.getsource(mtp_pipeline_stager)


@pytest.mark.parametrize("accepted,token_gear,request_gear,match", [
    ([0, 2, 0, 1], 8, 4, "between 0 and 1"),
    ([0, 1], 8, 4, "request order"),
    ([1, 1, 1, 1], 4, 4, "token gear"),
    ([0, 0, 0, 0], 8, 2, "request gear"),
])
def test_mtp_stager_rejects_invalid_layouts(
        accepted, token_gear, request_gear, match):
    with pytest.raises(ValueError, match=match):
        _stager().stage_drafter(
            _plan(), list("abcd"), accepted,
            token_gear=token_gear, request_gear=request_gear)
