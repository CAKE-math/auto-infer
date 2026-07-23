"""Persistent host/device input staging for fixed-size decode graph gears."""
from dataclasses import dataclass

import numpy as np
import torch

from auto_infer.engine.token_layout import slot_mapping
from auto_infer.worker.staging import (
    HostStaging, upload_dirty_block_table)


@dataclass(frozen=True)
class StagedDecodeInput:
    token_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    block_table: torch.Tensor
    kv_lengths: list[int]
    order: list[str]

    def data_ptrs(self) -> tuple[int, int, int, int]:
        return (
            self.token_ids.data_ptr(), self.positions.data_ptr(),
            self.slots.data_ptr(), self.block_table.data_ptr())


@dataclass(frozen=True)
class PreparedTokenSplice:
    groups: tuple
    fast_owner: object | None = None
    fast_count: int = 0


class DecodeInputStager:
    """Own pinned staging for one gear while preserving graph input addresses."""

    def __init__(self, *, tid, positions, slots, block_table,
                 block_size: int, scratch0: int, active_token_mask=None):
        self.tid = tid
        self.positions = positions
        self.slots = slots
        self.block_table = block_table
        self.active_token_mask = active_token_mask
        self.block_size = block_size
        self.scratch0 = scratch0
        self.gear = tid.shape[0]
        self.max_blocks = block_table.shape[1]
        host = HostStaging(tid.device.type != "cpu")
        self._tid_host, self._tid_np = host.allocate((self.gear,), torch.long)
        self._pos_host, self._pos_np = host.allocate((self.gear,), torch.long)
        self._slot_host, self._slot_np = host.allocate((self.gear,), torch.int32)
        if active_token_mask is not None:
            if (active_token_mask.dtype is not torch.bool
                    or tuple(active_token_mask.shape) != (self.gear,)):
                raise ValueError("decode active-token mask must be gear-shaped bool")
            self._active_host, self._active_np = host.allocate(
                (self.gear,), torch.bool)
        self._bt_host, self._bt_np = host.allocate(
            (self.gear, self.max_blocks), torch.int32)
        self._splice_src = torch.empty(
            self.gear, dtype=torch.long, device=tid.device)
        self._splice_dst = torch.empty(
            self.gear, dtype=torch.long, device=tid.device)
        self._splice_src_host, self._splice_src_np = host.allocate(
            (self.gear,), torch.long)
        self._splice_dst_host, self._splice_dst_np = host.allocate(
            (self.gear,), torch.long)
        self._non_blocking = host.non_blocking
        self._bt_shadow = np.full(
            (self.gear, self.max_blocks), np.iinfo(np.int32).min, dtype=np.int32)
        self.copied_block_rows = 0
        self.copied_block_elements = 0

    def stage(self, plan, scheduled=None) -> StagedDecodeInput:
        scheduled = plan.scheduled if scheduled is None else scheduled
        if len(scheduled) > self.gear:
            raise ValueError(
                f"decode batch {len(scheduled)} exceeds gear {self.gear}")

        self._bt_np.fill(0)
        if self.active_token_mask is not None:
            self._active_np.fill(False)
            self._active_np[:len(scheduled)] = True
        kv_lengths = []
        order = []
        for row, item in enumerate(scheduled):
            rid = item.request_id
            req = plan.get_request(rid)
            pos = req.num_computed_tokens
            blocks = plan.block_tables[rid]
            self._tid_np[row] = req.all_token_ids[pos]
            self._pos_np[row] = pos
            self._slot_np[row] = (
                blocks[pos // self.block_size] * self.block_size
                + pos % self.block_size)
            self._bt_np[row, :len(blocks)] = blocks
            kv_lengths.append(pos + 1)
            order.append(rid)

        for row in range(len(scheduled), self.gear):
            block = self.scratch0 + row
            self._tid_np[row] = 0
            self._pos_np[row] = 0
            self._slot_np[row] = block * self.block_size
            self._bt_np[row, 0] = block
            kv_lengths.append(1)

        self.tid.copy_(self._tid_host, non_blocking=self._non_blocking)
        self.positions.copy_(self._pos_host, non_blocking=self._non_blocking)
        self.slots.copy_(self._slot_host, non_blocking=self._non_blocking)
        if self.active_token_mask is not None:
            self.active_token_mask.copy_(
                self._active_host, non_blocking=self._non_blocking)

        rows, elements = upload_dirty_block_table(
            self.block_table, self._bt_host, self._bt_np, self._bt_shadow,
            self._non_blocking)
        self.copied_block_rows += rows
        self.copied_block_elements += elements

        return StagedDecodeInput(
            self.tid, self.positions, self.slots, self.block_table,
            kv_lengths, order)

    def prepare_splice(self, refs, request_order) -> PreparedTokenSplice:
        groups = {}
        for destination, request_id in enumerate(request_order):
            ref = refs.get(request_id)
            if ref is None:
                continue
            group = groups.setdefault(id(ref.owner), [ref.owner, [], []])
            group[1].append(destination)
            group[2].append(ref.row)
        values = tuple(groups.values())
        if len(values) == 1:
            owner, destinations, sources = values[0]
            count = len(destinations)
            if destinations == list(range(count)) and sources == list(range(count)):
                return PreparedTokenSplice((), owner, count)

        offset = 0
        prepared = []
        for owner, destinations, sources in values:
            end = offset + len(destinations)
            self._splice_dst_np[offset:end] = destinations
            self._splice_src_np[offset:end] = sources
            prepared.append((owner, offset, end))
            offset = end
        if offset:
            self._splice_src[:offset].copy_(
                self._splice_src_host[:offset],
                non_blocking=self._non_blocking)
            self._splice_dst[:offset].copy_(
                self._splice_dst_host[:offset],
                non_blocking=self._non_blocking)
        return PreparedTokenSplice(tuple(prepared))

    def apply_splice(self, prepared: PreparedTokenSplice) -> None:
        if prepared.fast_owner is not None:
            count = prepared.fast_count
            self.tid[:count].copy_(prepared.fast_owner.tokens[:count])
            return
        for owner, start, end in prepared.groups:
            values = owner.tokens.index_select(
                0, self._splice_src[start:end])
            self.tid.index_copy_(
                0, self._splice_dst[start:end], values)

    def splice(self, refs, request_order) -> None:
        self.apply_splice(self.prepare_splice(refs, request_order))


@dataclass(frozen=True)
class StagedContinuationInput:
    kv_lengths: list[int]


class ContinuationInputStager:
    """Persistent metadata staging for recurrent MTP continuation graphs."""

    def __init__(self, *, positions, slots, block_table,
                 block_size: int, scratch0: int):
        self.positions = positions
        self.slots = slots
        self.block_table = block_table
        self.block_size = block_size
        self.scratch0 = scratch0
        self.gear, self.max_blocks = block_table.shape
        self._batched_steps = positions.ndim == 2
        self.steps = positions.shape[0] if self._batched_steps else 1
        if slots.shape != positions.shape:
            raise ValueError("continuation positions and slots must match")
        host = HostStaging(positions.device.type != "cpu")
        self._pos_host, self._pos_np = host.allocate(
            tuple(positions.shape), torch.long)
        self._slot_host, self._slot_np = host.allocate(
            tuple(slots.shape), torch.int32)
        self._bt_host, self._bt_np = host.allocate(
            (self.gear, self.max_blocks), torch.int32)
        self._non_blocking = host.non_blocking
        self._bt_shadow = np.full(
            (self.gear, self.max_blocks), np.iinfo(np.int32).min,
            dtype=np.int32)
        self.copied_block_rows = 0
        self.copied_block_elements = 0

    def stage(self, plan, scheduled, accepted, step: int):
        if self._batched_steps:
            raise ValueError("use stage_all for a multi-step continuation gear")
        return StagedContinuationInput(
            self._stage_all(plan, scheduled, accepted, (step,))[0])

    def stage_all(self, plan, scheduled, accepted):
        return self._stage_all(
            plan, scheduled, accepted, range(1, self.steps + 1))

    def _stage_all(self, plan, scheduled, accepted, steps):
        if len(scheduled) > self.gear or len(accepted) != len(scheduled):
            raise ValueError("continuation metadata does not match graph gear")
        self._bt_np.fill(0)
        kv_by_step = []
        for step_index, step in enumerate(steps):
            kv_lengths = []
            pos_np = (self._pos_np[step_index] if self._batched_steps
                      else self._pos_np)
            slot_np = (self._slot_np[step_index] if self._batched_steps
                       else self._slot_np)
            for row, (item, accepted_count) in enumerate(zip(scheduled, accepted)):
                request = plan.get_request(item.request_id)
                position = request.num_computed_tokens + accepted_count + step
                blocks = plan.block_tables[item.request_id]
                pos_np[row] = position
                slot_np[row] = slot_mapping(blocks, position, self.block_size)
                self._bt_np[row, :len(blocks)] = blocks
                kv_lengths.append(position + 1)
            for row in range(len(scheduled), self.gear):
                block = self.scratch0 + row
                pos_np[row] = step
                slot_np[row] = block * self.block_size + step
                self._bt_np[row, 0] = block
                kv_lengths.append(step + 1)
            kv_by_step.append(kv_lengths)

        self.positions.copy_(self._pos_host, non_blocking=self._non_blocking)
        self.slots.copy_(self._slot_host, non_blocking=self._non_blocking)
        rows, elements = upload_dirty_block_table(
            self.block_table, self._bt_host, self._bt_np, self._bt_shadow,
            self._non_blocking)
        self.copied_block_rows += rows
        self.copied_block_elements += elements
        return kv_by_step


class SpecDecodeInputStager:
    """Persistent pinned staging for a speculative decode graph gear."""

    def __init__(self, *, tid, positions, slots, block_table, drafts,
                 active_mask=None, active_token_mask=None,
                 block_size: int, scratch0: int, geometry):
        self.tid = tid
        self.positions = positions
        self.slots = slots
        self.block_table = block_table
        self.drafts = drafts
        self.active_mask = active_mask
        self.active_token_mask = active_token_mask
        self.geometry = geometry
        self.query_width = geometry.query_width
        self.block_size = block_size
        self.scratch0 = scratch0
        self.gear = block_table.shape[0]
        self.max_blocks = block_table.shape[1]
        if tid.numel() != self.gear * self.query_width:
            raise ValueError(
                "spec decode token buffer must match the configured query width")
        host = HostStaging(tid.device.type != "cpu")
        query_rows = self.gear * self.query_width
        self._tid_host, self._tid_np = host.allocate((query_rows,), torch.long)
        self._pos_host, self._pos_np = host.allocate((query_rows,), torch.long)
        self._slot_host, self._slot_np = host.allocate((query_rows,), torch.int32)
        self._draft_host, self._draft_np = host.allocate(
            tuple(drafts.shape), torch.long)
        if active_mask is not None:
            if tuple(active_mask.shape) != (self.gear,):
                raise ValueError("spec active mask must have one row per request")
            self._active_host, self._active_np = host.allocate(
                (self.gear,), torch.int32)
        if active_token_mask is not None:
            if (active_token_mask.dtype is not torch.bool
                    or tuple(active_token_mask.shape) != (query_rows,)):
                raise ValueError(
                    "spec EP active-token mask must match query rows and use bool")
            self._ep_active_host, self._ep_active_np = host.allocate(
                (query_rows,), torch.bool)
        self._bt_host, self._bt_np = host.allocate(
            (self.gear, self.max_blocks), torch.int32)
        self._non_blocking = host.non_blocking
        self._bt_shadow = np.full(
            (self.gear, self.max_blocks), np.iinfo(np.int32).min,
            dtype=np.int32)
        self.copied_block_rows = 0
        self.copied_block_elements = 0

    def stage(self, plan, scheduled=None) -> StagedDecodeInput:
        scheduled = plan.scheduled if scheduled is None else scheduled
        if len(scheduled) > self.gear:
            raise ValueError(
                f"spec decode batch {len(scheduled)} exceeds gear {self.gear}")

        self._bt_np.fill(0)
        if self.active_mask is not None:
            self._active_np.fill(0)
        if self.active_token_mask is not None:
            self._ep_active_np.fill(False)
            self._ep_active_np[:len(scheduled) * self.query_width] = True
        kv_lengths = []
        order = []
        for row, item in enumerate(scheduled):
            rid = item.request_id
            req = plan.get_request(rid)
            pos = req.num_computed_tokens
            blocks = plan.block_tables[rid]
            if len(req.spec_draft) != self.geometry.draft_depth:
                raise ValueError(
                    f"request {rid} has {len(req.spec_draft)} drafts; "
                    f"expected {self.geometry.draft_depth}")
            start = row * self.query_width
            self._tid_np[start:start + self.query_width] = (
                req.all_token_ids[pos], *req.spec_draft)
            self._draft_np[row] = req.spec_draft
            if self.active_mask is not None:
                self._active_np[row] = 1
            for offset in range(self.query_width):
                query_pos = pos + offset
                self._pos_np[start + offset] = query_pos
                self._slot_np[start + offset] = slot_mapping(
                    blocks, query_pos, self.block_size)
            self._bt_np[row, :len(blocks)] = blocks
            kv_lengths.append(pos + self.query_width)
            order.append(rid)

        for row in range(len(scheduled), self.gear):
            block = self.scratch0 + row
            start = row * self.query_width
            self._tid_np[start:start + self.query_width] = 0
            self._draft_np[row] = 0
            self._pos_np[start:start + self.query_width] = np.arange(
                self.query_width)
            self._slot_np[start:start + self.query_width] = (
                block * self.block_size + np.arange(self.query_width))
            self._bt_np[row, 0] = block
            kv_lengths.append(self.query_width)

        self.tid.copy_(self._tid_host, non_blocking=self._non_blocking)
        self.positions.copy_(self._pos_host, non_blocking=self._non_blocking)
        self.slots.copy_(self._slot_host, non_blocking=self._non_blocking)
        self.drafts.copy_(self._draft_host, non_blocking=self._non_blocking)
        if self.active_mask is not None:
            self.active_mask.copy_(
                self._active_host, non_blocking=self._non_blocking)
        if self.active_token_mask is not None:
            self.active_token_mask.copy_(
                self._ep_active_host, non_blocking=self._non_blocking)

        rows, elements = upload_dirty_block_table(
            self.block_table, self._bt_host, self._bt_np, self._bt_shadow,
            self._non_blocking)
        self.copied_block_rows += rows
        self.copied_block_elements += elements

        return StagedDecodeInput(
            self.tid, self.positions, self.slots, self.block_table,
            kv_lengths, order)
