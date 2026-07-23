import pytest
import torch

from auto_infer.worker.async_slots import DeviceTokenStore, ExecutionSlotPool


def test_slot_pool_never_reuses_an_inflight_slot():
    pool = ExecutionSlotPool(2)

    first = pool.acquire()
    second = pool.acquire()

    assert first.slot_id != second.slot_id
    with pytest.raises(RuntimeError, match="no free async execution slot"):
        pool.acquire()

    pool.release(first)
    assert pool.acquire().slot_id == first.slot_id


def test_slot_pool_rejects_foreign_or_double_release():
    pool = ExecutionSlotPool(1)
    slot = pool.acquire()
    pool.release(slot)

    with pytest.raises(RuntimeError, match="not leased"):
        pool.release(slot)


def test_device_token_store_retains_skipped_request_and_reuses_released_row():
    store = DeviceTokenStore(2, device=torch.device("cpu"))
    store.write(torch.tensor([11, 22]), ("a", "b"))
    first = store.refs(("a", "b"))

    store.write(torch.tensor([12]), ("a",))
    second = store.refs(("a", "b"))

    assert int(second["a"].owner.tokens[second["a"].row]) == 12
    assert int(second["b"].owner.tokens[second["b"].row]) == 22
    assert first["b"].row == second["b"].row

    released_row = second["a"].row
    store.release(("a",))
    store.write(torch.tensor([33]), ("c",))
    assert store.refs(("c",))["c"].row == released_row


def test_device_token_store_has_fixed_capacity():
    store = DeviceTokenStore(1, device=torch.device("cpu"))
    store.write(torch.tensor([1]), ("a",))

    with pytest.raises(RuntimeError, match="capacity"):
        store.write(torch.tensor([2]), ("b",))
