from types import SimpleNamespace

import torch

from auto_infer.worker.prefill_input_stager import PrefillInputStager


class _Plan:
    def __init__(self, scheduled, requests, block_tables):
        self.scheduled = tuple(scheduled)
        self.requests = requests
        self.block_tables = block_tables

    def get_request(self, request_id):
        return self.requests[request_id]


def _item(request_id, count):
    return SimpleNamespace(request_id=request_id, num_tokens_to_compute=count)


def _request(tokens, computed, prefill):
    return SimpleNamespace(
        all_token_ids=tuple(tokens), num_computed_tokens=computed,
        num_prefill_tokens=prefill)


def _stager(query_gear=6, sequence_gear=2, active_token_mask=None):
    return PrefillInputStager(
        token_ids=torch.zeros(query_gear, dtype=torch.long),
        positions=torch.zeros(query_gear, dtype=torch.long),
        slots=torch.zeros(query_gear, dtype=torch.int32),
        block_table=torch.zeros(sequence_gear, 4, dtype=torch.int32),
        sample_rows=torch.zeros(sequence_gear, dtype=torch.long),
        active_token_mask=active_token_mask,
        block_size=4, scratch0=100)


def test_prefill_stager_builds_exact_shape_tnd_metadata():
    plan = _Plan(
        [_item("a", 4), _item("b", 2)],
        {
            "a": _request([10, 11, 12, 13], 0, 4),
            "b": _request([20, 21, 22], 1, 3),
        },
        {"a": (5,), "b": (7,)})

    staged = _stager().stage(plan)

    assert staged.query_tokens == 6
    assert staged.token_ids.tolist() == [10, 11, 12, 13, 21, 22]
    assert staged.positions.tolist() == [0, 1, 2, 3, 1, 2]
    assert staged.slots.tolist() == [20, 21, 22, 23, 29, 30]
    assert staged.cumulative_query_lengths == [4, 6]
    assert staged.kv_lengths == [4, 3]
    assert staged.sample_order == ["a", "b"]
    assert staged.sample_rows[:2].tolist() == [3, 5]
    assert staged.splice_order == ["a", "b"]
    assert staged.splice_rows == [3, 5]
    assert staged.block_table[0, 0].item() == 5
    assert staged.block_table[1, 0].item() == 7


def test_prefill_stager_preserves_device_addresses_across_steps():
    stager = _stager(query_gear=4, sequence_gear=1)
    plan = _Plan(
        [_item("a", 4)], {"a": _request([1, 2, 3, 4], 0, 4)},
        {"a": (2,)})

    first = stager.stage(plan)
    pointers = first.data_ptrs()
    second = stager.stage(plan)

    assert second.data_ptrs() == pointers


def test_prefill_stager_pads_to_token_gear_without_sampling_padding():
    stager = _stager(query_gear=8, sequence_gear=8)
    plan = _Plan(
        [_item("a", 3), _item("b", 2)],
        {
            "a": _request([10, 11, 12], 0, 3),
            "b": _request([20, 21], 0, 2),
        },
        {"a": (5,), "b": (7,)})

    staged = stager.stage(plan)

    assert staged.real_query_tokens == 5
    assert staged.query_tokens == 8
    assert staged.sequence_count == 2
    assert tuple(staged.block_table.shape) == (3, 4)
    assert staged.sample_rows[:2].tolist() == [2, 4]
    assert staged.cumulative_query_lengths == [3, 5, 8]
    assert staged.kv_lengths == [3, 2, 3]
    assert staged.sample_order == ["a", "b"]
    assert all(slot >= 100 * 4 for slot in staged.slots[5:].tolist())


def test_prefill_stager_updates_fixed_address_live_token_mask():
    mask = torch.zeros(8, dtype=torch.bool)
    stager = _stager(
        query_gear=8, sequence_gear=8, active_token_mask=mask)
    pointer = mask.data_ptr()
    plan = _Plan(
        [_item("a", 3), _item("b", 2)],
        {"a": _request([10, 11, 12], 0, 3),
         "b": _request([20, 21], 0, 2)},
        {"a": (5,), "b": (7,)})

    stager.stage(plan)

    assert mask.tolist() == [True] * 5 + [False] * 3
    assert mask.data_ptr() == pointer


def test_prefill_stager_reuses_token_gear_across_sequence_counts():
    stager = _stager(query_gear=8, sequence_gear=8)
    one = stager.stage(_Plan(
        [_item("a", 8)], {"a": _request(range(8), 0, 8)},
        {"a": (1, 2)}))
    one_ptr = one.block_table.data_ptr()
    two = stager.stage(_Plan(
        [_item("a", 4), _item("b", 4)],
        {"a": _request(range(4), 0, 4),
         "b": _request(range(10, 14), 0, 4)},
        {"a": (1,), "b": (2,)}))

    assert one.block_table.shape[0] == 1
    assert two.block_table.shape[0] == 2
    assert one_ptr == two.block_table.data_ptr()


def test_prefill_stager_scratch_fills_block_table_for_padding():
    stager = _stager(query_gear=8, sequence_gear=8)
    staged = stager.stage(_Plan(
        [_item("a", 3)], {"a": _request([1, 2, 3], 0, 3)},
        {"a": (9,)}))

    assert staged.block_table[0, 0].item() == 9
    assert staged.block_table[1, 0].item() == 100
    assert staged.block_table[1, 1].item() == 101
    assert all(slot >= 100 * 4 for slot in staged.slots[3:].tolist())


def test_prefill_padding_stays_inside_reserved_scratch_near_model_limit():
    query_gear = 8
    stager = PrefillInputStager(
        token_ids=torch.zeros(query_gear, dtype=torch.long),
        positions=torch.zeros(query_gear, dtype=torch.long),
        slots=torch.zeros(query_gear, dtype=torch.int32),
        block_table=torch.zeros(query_gear, 32, dtype=torch.int32),
        sample_rows=torch.zeros(query_gear, dtype=torch.long),
        block_size=16, scratch0=100)
    staged = stager.stage(_Plan(
        [_item("a", 1)],
        {"a": _request(range(497), 496, 497)},
        {"a": tuple(range(32))}))

    padding_blocks = [slot // 16 for slot in staged.slots[1:].tolist()]
    assert all(100 <= block < 100 + query_gear for block in padding_blocks)
    assert staged.cumulative_query_lengths == [1, 8]
    assert staged.kv_lengths == [497, 7]
    assert staged.block_table.shape[0] == 2


def test_incomplete_chunked_prefill_with_padding_does_not_sample():
    stager = _stager(query_gear=8, sequence_gear=8)
    staged = stager.stage(_Plan(
        [_item("a", 3)],
        {"a": _request(range(10), 0, 10)},
        {"a": (9,)}))

    assert staged.real_query_tokens == 3
    assert staged.cumulative_query_lengths == [3, 8]
    assert staged.kv_lengths == [3, 5]
    assert staged.sample_order == []
    assert staged.sample_rows.tolist() == [0] * 8
    assert staged.block_table.shape[0] == 2


def test_prefill_stager_rejects_query_or_sequence_overflow():
    stager = _stager(query_gear=4, sequence_gear=1)
    too_many_tokens = _Plan(
        [_item("a", 5)], {"a": _request(range(5), 0, 5)}, {"a": (1, 2)})
    too_many_sequences = _Plan(
        [_item("a", 1), _item("b", 1)],
        {"a": _request([1], 0, 1), "b": _request([2], 0, 1)},
        {"a": (1,), "b": (2,)})

    for plan in (too_many_tokens, too_many_sequences):
        try:
            stager.stage(plan)
        except ValueError as exc:
            assert "exceeds prefill gear" in str(exc)
        else:
            raise AssertionError("expected prefill gear overflow")
