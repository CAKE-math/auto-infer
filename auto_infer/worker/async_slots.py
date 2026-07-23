"""Ownership primitives for zero-host-bubble graph decode."""
from collections import deque
from dataclasses import dataclass

import torch

from auto_infer.engine.execution import DeviceTokenBatch


@dataclass
class ExecutionSlot:
    slot_id: int
    sequence: int = -1
    leased: bool = False


class ExecutionSlotPool:
    """Fixed-depth pool; a slot cannot be reused while a submission owns it."""

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError("async execution slot depth must be positive")
        self.slots = tuple(ExecutionSlot(index) for index in range(depth))
        self._free = deque(self.slots)
        self._next_sequence = 0

    @property
    def available(self) -> int:
        return len(self._free)

    def acquire(self) -> ExecutionSlot:
        if not self._free:
            raise RuntimeError("no free async execution slot")
        slot = self._free.popleft()
        slot.leased = True
        slot.sequence = self._next_sequence
        self._next_sequence += 1
        return slot

    def release(self, slot: ExecutionSlot) -> None:
        if slot not in self.slots or not slot.leased:
            raise RuntimeError("async execution slot is not leased")
        slot.leased = False
        self._free.append(slot)


class DeviceTokenStore:
    """Stable device rows keyed by request ID.

    Graph outputs are scattered here on the producer stream. DeviceTokenRef
    objects may therefore survive graph-output-slot reuse and skipped batches.
    """

    def __init__(self, capacity: int, device, dtype=torch.long):
        if capacity < 1:
            raise ValueError("device token store capacity must be positive")
        self.tokens = torch.empty(capacity, dtype=dtype, device=device)
        self._free = deque(range(capacity))
        self._row_by_request: dict[str, int] = {}

    def _rows_for(self, order) -> tuple[int, ...]:
        rows = []
        allocated = []
        try:
            for rid in order:
                row = self._row_by_request.get(rid)
                if row is None:
                    if not self._free:
                        raise RuntimeError("device token store capacity exhausted")
                    row = self._free.popleft()
                    self._row_by_request[rid] = row
                    allocated.append((rid, row))
                rows.append(row)
        except Exception:
            for rid, row in reversed(allocated):
                self._row_by_request.pop(rid, None)
                self._free.appendleft(row)
            raise
        return tuple(rows)

    def reserve(self, order) -> tuple[int, ...]:
        return self._rows_for(tuple(order))

    def commit(self, sampled, order, rows, indices) -> DeviceTokenBatch:
        order = tuple(order)
        rows = tuple(rows)
        if sampled.ndim != 1 or sampled.shape[0] != len(order):
            raise ValueError("sampled tokens and request order must have equal length")
        if len(rows) != len(order) or indices.numel() < len(rows):
            raise ValueError("token-store rows do not match sampled order")
        self.tokens.index_copy_(0, indices[:len(rows)], sampled)
        return DeviceTokenBatch.from_rows(self.tokens, order, rows)

    def write(self, sampled: torch.Tensor, order, row_buffer=None) -> DeviceTokenBatch:
        order = tuple(order)
        if sampled.ndim != 1 or sampled.shape[0] != len(order):
            raise ValueError("sampled tokens and request order must have equal length")
        rows = self.reserve(order)
        if row_buffer is None:
            indices = torch.tensor(rows, dtype=torch.long, device=sampled.device)
        else:
            if row_buffer.dtype is not torch.long or row_buffer.numel() < len(rows):
                raise ValueError("token-store row buffer must be a sufficiently large int64 tensor")
            indices = row_buffer[:len(rows)]
            indices.copy_(torch.tensor(rows, dtype=torch.long, device=indices.device))
        return self.commit(sampled, order, rows, indices)

    def refs(self, order) -> dict:
        order = tuple(order)
        rows = []
        for rid in order:
            if rid not in self._row_by_request:
                raise KeyError(rid)
            rows.append(self._row_by_request[rid])
        return DeviceTokenBatch.from_rows(self.tokens, order, rows).refs()

    def release(self, request_ids) -> None:
        for rid in request_ids:
            row = self._row_by_request.pop(rid, None)
            if row is not None:
                self._free.append(row)
