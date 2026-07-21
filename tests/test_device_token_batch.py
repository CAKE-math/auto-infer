import torch

from auto_infer.engine.execution import DeviceTokenBatch
from auto_infer.worker.staging import splice_device_tokens


def test_sampled_batch_retains_one_tensor_without_per_row_clones():
    tokens = torch.tensor([3, 5, 7])

    batch = DeviceTokenBatch.from_output(tokens, ["a", "b", "c"])

    assert batch.tokens.data_ptr() == tokens.data_ptr()
    assert [batch.row_by_request[rid] for rid in ("c", "a")] == [2, 0]
    refs = batch.refs()
    assert refs["a"].owner is batch and refs["a"].row == 0
    assert refs["c"].owner is batch and refs["c"].row == 2


def test_batch_metadata_is_immutable_and_rejects_duplicate_ids():
    tokens = torch.tensor([3, 5])
    batch = DeviceTokenBatch.from_output(tokens, ["a", "b"])

    try:
        batch.row_by_request["a"] = 1
    except TypeError:
        pass
    else:
        raise AssertionError("row mapping must be immutable")

    try:
        DeviceTokenBatch.from_output(tokens, ["a", "a"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate request IDs must be rejected")


def test_async_splice_groups_rows_by_owner_without_scalar_clones():
    first = DeviceTokenBatch.from_output(torch.tensor([11, 13, 17]), ["a", "b", "c"])
    second = DeviceTokenBatch.from_output(torch.tensor([19, 23]), ["d", "e"])
    refs = {**first.refs(), **second.refs()}
    target = torch.zeros(6, dtype=torch.long)
    splice_device_tokens(target, [4, 1, 5], ["c", "a", "e"], refs)

    assert target.tolist() == [0, 11, 0, 0, 17, 23]
