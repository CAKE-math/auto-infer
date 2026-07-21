"""Event-backed device-to-pinned-host token copies for async output."""
from threading import Lock

import torch

from auto_infer.engine.execution import ExecutionResult


class PinnedTokenBufferPool:
    """Grow-on-demand pool; buffers are reused after their copy is consumed."""

    def __init__(self, pin_memory: bool = True):
        self.pin_memory = pin_memory
        self._free = []
        self._lock = Lock()

    @property
    def available(self) -> int:
        with self._lock:
            return len(self._free)

    def acquire(self, count: int) -> torch.Tensor:
        with self._lock:
            for index, tensor in enumerate(self._free):
                if tensor.numel() >= count:
                    return self._free.pop(index)
        tensor = torch.empty(count, dtype=torch.long)
        if self.pin_memory:
            try:
                tensor = tensor.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass
        return tensor

    def release(self, tensor: torch.Tensor) -> None:
        with self._lock:
            self._free.append(tensor)


class AsyncHostCopy:
    """Own a host buffer until its producer event completes and values map."""

    def __init__(self, cpu_tokens, count, order, ready_event, release):
        self.cpu_tokens = cpu_tokens
        self.count = count
        self.order = tuple(order)
        self.ready_event = ready_event
        self._release = release
        self._result = None
        self._lock = Lock()

    def result(self) -> ExecutionResult:
        with self._lock:
            if self._result is not None:
                return self._result
            if self.ready_event is not None:
                self.ready_event.synchronize()
            values = self.cpu_tokens[:self.count].tolist()
            self._result = ExecutionResult.from_single_tokens(
                {rid: int(value) for rid, value in zip(self.order, values)})
            self._release(self.cpu_tokens)
            return self._result


def enqueue_host_copy(tokens, order, copy_stream, pool, *, protect_source=False):
    """Queue D2H now; the output thread later waits/maps CPU-resident values."""
    count = tokens.shape[0]
    cpu_tokens = pool.acquire(count)
    if tokens.device.type == "cpu":
        cpu_tokens[:count].copy_(tokens)
        return AsyncHostCopy(cpu_tokens, count, order, None, pool.release)

    producer_stream = torch.npu.current_stream()
    produced = torch.npu.Event()
    produced.record(producer_stream)
    with torch.npu.stream(copy_stream):
        produced.wait(copy_stream)
        cpu_tokens[:count].copy_(tokens, non_blocking=True)
        ready = torch.npu.Event()
        ready.record(copy_stream)
    if protect_source:
        # Captured output buffers are reused by the next replay. Prevent that
        # replay from overwriting tokens until the tiny D2H has consumed them.
        ready.wait(producer_stream)
    return AsyncHostCopy(cpu_tokens, count, order, ready, pool.release)
