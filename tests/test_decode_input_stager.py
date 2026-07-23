import torch

from auto_infer.engine.execution import DeviceTokenBatch
from auto_infer.spec_decode.geometry import MtpGeometry
from auto_infer.worker.decode_input_stager import (
    ContinuationInputStager, DecodeInputStager, SpecDecodeInputStager)


class _Req:
    def __init__(self, tokens, computed, draft=99):
        self.all_token_ids = tuple(tokens)
        self.num_computed_tokens = computed
        self.spec_draft = list(draft) if isinstance(draft, (list, tuple)) else [draft]


class _Scheduled:
    def __init__(self, rid):
        self.request_id = rid


class _Plan:
    def __init__(self, rows):
        self.scheduled = tuple(_Scheduled(rid) for rid in rows)
        self.requests = {rid: req for rid, (req, _) in rows.items()}
        self.block_tables = {rid: tuple(bt) for rid, (_, bt) in rows.items()}

    def get_request(self, rid):
        return self.requests[rid]


def _stager(gear=4, max_blocks=4, active_token_mask=None):
    return DecodeInputStager(
        tid=torch.zeros(gear, dtype=torch.long),
        positions=torch.zeros(gear, dtype=torch.long),
        slots=torch.zeros(gear, dtype=torch.int32),
        block_table=torch.zeros(gear, max_blocks, dtype=torch.int32),
        active_token_mask=active_token_mask,
        block_size=4,
        scratch0=100,
    )


def _plan(computed_a=5, include_b=True):
    rows = {
        "a": (_Req(range(20), computed_a), [7, 8, 9]),
    }
    if include_b:
        rows["b"] = (_Req(range(20, 40), 1), [3])
    return _Plan(rows)


def test_stager_reuses_device_addresses_across_steps():
    stager = _stager()
    first = stager.stage(_plan())
    addresses = first.data_ptrs()

    second = stager.stage(_plan(computed_a=6))

    assert second.data_ptrs() == addresses


def test_unchanged_block_rows_are_not_copied_twice():
    stager = _stager()
    stager.stage(_plan())
    copied = stager.copied_block_rows

    stager.stage(_plan(computed_a=6))

    assert stager.copied_block_rows == copied


def test_live_and_padding_rows_match_decode_contract():
    stager = _stager()

    staged = stager.stage(_plan())

    assert staged.order == ["a", "b"]
    assert staged.kv_lengths == [6, 2, 1, 1]
    assert staged.token_ids.tolist() == [5, 21, 0, 0]
    assert staged.positions.tolist() == [5, 1, 0, 0]
    assert staged.slots.tolist() == [8 * 4 + 1, 3 * 4 + 1, 102 * 4, 103 * 4]
    assert staged.block_table[0].tolist() == [7, 8, 9, 0]
    assert staged.block_table[1].tolist() == [3, 0, 0, 0]
    assert staged.block_table[2].tolist() == [102, 0, 0, 0]
    assert staged.block_table[3].tolist() == [103, 0, 0, 0]


def test_decode_stager_updates_fixed_address_live_token_mask():
    mask = torch.zeros(4, dtype=torch.bool)
    stager = _stager(active_token_mask=mask)
    pointer = mask.data_ptr()

    stager.stage(_plan(include_b=True))
    assert mask.tolist() == [True, True, False, False]
    stager.stage(_plan(include_b=False))

    assert mask.tolist() == [True, False, False, False]
    assert mask.data_ptr() == pointer


def test_row_transitioning_to_padding_is_copied_once():
    stager = _stager()
    stager.stage(_plan())
    copied = stager.copied_block_rows

    staged = stager.stage(_plan(computed_a=6, include_b=False))

    assert stager.copied_block_rows == copied + 1
    assert staged.block_table[1].tolist() == [101, 0, 0, 0]


def test_decode_splice_fast_path_uses_aligned_batch_without_index_tensors():
    stager = _stager()
    owner = DeviceTokenBatch.from_output(
        torch.tensor([31, 37]), ("a", "b"))

    prepared = stager.prepare_splice(
        owner.refs(), ["a", "b"])
    stager.apply_splice(prepared)

    assert prepared.fast_owner is owner
    assert stager.tid[:2].tolist() == [31, 37]


def test_decode_splice_persistent_indices_cover_skips_and_reordering():
    stager = _stager()
    first = DeviceTokenBatch.from_output(
        torch.tensor([11, 13]), ("a", "b"))
    second = DeviceTokenBatch.from_output(
        torch.tensor([17]), ("c",))
    refs = {**first.refs(), **second.refs()}

    prepared = stager.prepare_splice(refs, ["c", "a"])
    stager.apply_splice(prepared)

    assert prepared.fast_owner is None
    assert stager.tid[:2].tolist() == [17, 11]


def test_spec_stager_persists_two_token_rows_and_only_copies_dirty_blocks():
    gear = 4
    active_mask = torch.zeros(gear, dtype=torch.int32)
    stager = SpecDecodeInputStager(
        tid=torch.zeros(gear * 2, dtype=torch.long),
        positions=torch.zeros(gear * 2, dtype=torch.long),
        slots=torch.zeros(gear * 2, dtype=torch.int32),
        block_table=torch.zeros(gear, 4, dtype=torch.int32),
        drafts=torch.zeros(gear, 1, dtype=torch.long),
        active_mask=active_mask,
        block_size=4,
        scratch0=100,
        geometry=MtpGeometry(1),
    )

    first = stager.stage(_plan())
    assert first.order == ["a", "b"]
    assert first.kv_lengths == [7, 3, 2, 2]
    assert first.token_ids.tolist() == [5, 99, 21, 99, 0, 0, 0, 0]
    assert first.positions.tolist() == [5, 6, 1, 2, 0, 1, 0, 1]
    assert first.slots.tolist() == [33, 34, 13, 14, 408, 409, 412, 413]
    assert active_mask.tolist() == [1, 1, 0, 0]
    copied = stager.copied_block_rows
    addresses = first.data_ptrs()
    second = stager.stage(_plan(computed_a=6))

    assert second.data_ptrs() == addresses
    assert active_mask.tolist() == [1, 1, 0, 0]
    assert stager.copied_block_rows == copied


def test_spec_stager_fills_every_multi_step_query_row():
    geometry = MtpGeometry(3)
    stager = SpecDecodeInputStager(
        tid=torch.zeros(8, dtype=torch.long),
        positions=torch.zeros(8, dtype=torch.long),
        slots=torch.zeros(8, dtype=torch.int32),
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        drafts=torch.zeros(2, 3, dtype=torch.long),
        block_size=4, scratch0=100, geometry=geometry)
    plan = _Plan({"a": (_Req(range(20), 5, [90, 91, 92]), [7, 8, 9])})

    staged = stager.stage(plan)

    assert staged.token_ids.tolist() == [5, 90, 91, 92, 0, 0, 0, 0]
    assert staged.positions.tolist() == [5, 6, 7, 8, 0, 1, 2, 3]
    assert stager.drafts.tolist() == [[90, 91, 92], [0, 0, 0]]


def test_spec_request_mask_expands_to_ep_token_rows():
    geometry = MtpGeometry(3)
    ep_mask = torch.zeros(8, dtype=torch.bool)
    stager = SpecDecodeInputStager(
        tid=torch.zeros(8, dtype=torch.long),
        positions=torch.zeros(8, dtype=torch.long),
        slots=torch.zeros(8, dtype=torch.int32),
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        drafts=torch.zeros(2, 3, dtype=torch.long),
        active_token_mask=ep_mask,
        block_size=4, scratch0=100, geometry=geometry)

    stager.stage(_Plan({
        "a": (_Req(range(20), 5, [90, 91, 92]), [7, 8, 9])}))

    assert ep_mask.tolist() == [True, True, True, True,
                                False, False, False, False]


def test_continuation_stager_reuses_buffers_and_skips_clean_block_rows():
    positions = torch.zeros(2, dtype=torch.long)
    slots = torch.zeros(2, dtype=torch.int32)
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    stager = ContinuationInputStager(
        positions=positions, slots=slots, block_table=block_table,
        block_size=4, scratch0=100)
    plan = _Plan({"a": (_Req(range(20), 5, [90, 91]), [7, 8, 9])})

    first = stager.stage(plan, plan.scheduled, accepted=[1], step=1)
    copied = stager.copied_block_rows
    copied_elements = stager.copied_block_elements
    addresses = (positions.data_ptr(), slots.data_ptr(), block_table.data_ptr())
    second = stager.stage(plan, plan.scheduled, accepted=[2], step=2)

    assert (positions.data_ptr(), slots.data_ptr(), block_table.data_ptr()) == addresses
    assert first.kv_lengths == [8, 2]
    assert second.kv_lengths == [10, 3]
    assert positions.tolist() == [9, 2]
    assert slots.tolist() == [9 * 4 + 1, 101 * 4 + 2]
    assert stager.copied_block_rows == copied
    assert copied_elements == copied * stager.max_blocks
    assert stager.copied_block_elements == copied_elements


def test_continuation_stager_counts_two_dirty_block_rows_and_elements():
    stager = ContinuationInputStager(
        positions=torch.zeros(2, dtype=torch.long),
        slots=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        block_size=4, scratch0=100)
    first = _Plan({
        "a": (_Req(range(20), 5, [90, 91]), [7, 8, 9]),
        "b": (_Req(range(20, 40), 1, [80, 81]), [3]),
    })
    second = _Plan({
        "a": (first.get_request("a"), [7, 8, 10]),
        "b": (first.get_request("b"), [4]),
    })
    stager.stage(first, first.scheduled, accepted=[1, 1], step=1)
    rows_before = stager.copied_block_rows
    elements_before = stager.copied_block_elements

    stager.stage(second, second.scheduled, accepted=[1, 1], step=1)

    assert stager.copied_block_rows - rows_before == 2
    assert stager.copied_block_elements - elements_before == 8


def test_continuation_stager_stages_entire_depth_in_one_copy():
    positions = torch.zeros(2, 2, dtype=torch.long)
    slots = torch.zeros(2, 2, dtype=torch.int32)
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    stager = ContinuationInputStager(
        positions=positions, slots=slots, block_table=block_table,
        block_size=4, scratch0=100)
    plan = _Plan({"a": (_Req(range(20), 5, [90, 91, 92]), [7, 8, 9])})

    kv_by_step = stager.stage_all(plan, plan.scheduled, accepted=[1])

    assert kv_by_step == [[8, 2], [9, 3]]
    assert positions.tolist() == [[7, 1], [8, 2]]
