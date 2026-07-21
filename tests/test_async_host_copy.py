import torch

from auto_infer.worker.async_output import AsyncHostCopy, PinnedTokenBufferPool


class _Event:
    def __init__(self, trace):
        self.trace = trace

    def synchronize(self):
        self.trace.append("wait")


def test_async_host_copy_waits_before_read_and_releases_buffer():
    trace = []
    pool = PinnedTokenBufferPool(pin_memory=False)
    buffer = pool.acquire(3)
    buffer[:3].copy_(torch.tensor([2, 4, 6]))
    copy = AsyncHostCopy(
        cpu_tokens=buffer, count=3, order=("a", "b", "c"),
        ready_event=_Event(trace), release=pool.release)

    result = copy.result()

    assert trace == ["wait"]
    assert result.single_tokens() == {"a": 2, "b": 4, "c": 6}
    assert pool.available == 1


def test_host_copy_result_is_idempotent():
    pool = PinnedTokenBufferPool(pin_memory=False)
    buffer = pool.acquire(1)
    buffer[0] = 9
    copy = AsyncHostCopy(buffer, 1, ("r",), None, pool.release)

    assert copy.result().single_tokens() == {"r": 9}
    assert copy.result().single_tokens() == {"r": 9}
    assert pool.available == 1
