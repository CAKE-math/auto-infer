"""Persistent fixed-address staging for prefill and mixed ACL graph gears."""
from dataclasses import dataclass

import numpy as np
import torch

from auto_infer.engine.token_layout import slot_mapping
from auto_infer.worker.staging import (
    HostStaging, splice_device_tokens, upload_dirty_block_table)


@dataclass(frozen=True)
class StagedPrefillInput:
    token_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    block_table: torch.Tensor
    sample_rows: torch.Tensor
    cumulative_query_lengths: list[int]
    kv_lengths: list[int]
    sample_order: list[str]
    splice_order: list[str]
    splice_rows: list[int]
    query_tokens: int
    real_query_tokens: int
    sequence_count: int

    def data_ptrs(self) -> tuple[int, int, int, int, int]:
        return (
            self.token_ids.data_ptr(), self.positions.data_ptr(),
            self.slots.data_ptr(), self.block_table.data_ptr(),
            self.sample_rows.data_ptr())


class PrefillInputStager:
    """Own host/device buffers for one ``(query_gear, sequence_gear)``."""

    def __init__(self, *, token_ids, positions, slots, block_table, sample_rows,
                 block_size: int, scratch0: int, active_token_mask=None):
        self.token_ids = token_ids
        self.positions = positions
        self.slots = slots
        self.block_table = block_table
        self.sample_rows = sample_rows
        self.active_token_mask = active_token_mask
        self.block_size = block_size
        self.scratch0 = scratch0
        self.query_gear = token_ids.shape[0]
        self.sequence_gear = sample_rows.shape[0]
        self.max_blocks = block_table.shape[1]
        host = HostStaging(token_ids.device.type != "cpu")
        self._tid_host, self._tid_np = host.allocate((self.query_gear,), torch.long)
        self._pos_host, self._pos_np = host.allocate((self.query_gear,), torch.long)
        self._slot_host, self._slot_np = host.allocate((self.query_gear,), torch.int32)
        self._bt_host, self._bt_np = host.allocate(
            tuple(block_table.shape), torch.int32)
        self._sample_host, self._sample_np = host.allocate(
            (self.sequence_gear,), torch.long)
        if active_token_mask is not None:
            if (active_token_mask.dtype is not torch.bool
                    or tuple(active_token_mask.shape) != (self.query_gear,)):
                raise ValueError("prefill active-token mask must be gear-shaped bool")
            self._active_host, self._active_np = host.allocate(
                (self.query_gear,), torch.bool)
        self._non_blocking = host.non_blocking
        self._bt_shadow = np.full(
            tuple(block_table.shape), np.iinfo(np.int32).min,
            dtype=np.int32)
        self.copied_block_rows = 0
        self.copied_block_elements = 0

    def stage(self, plan) -> StagedPrefillInput:
        scheduled = plan.scheduled
        real_query_tokens = sum(
            item.num_tokens_to_compute for item in scheduled)
        sequence_count = len(scheduled)
        if (real_query_tokens > self.query_gear
                or sequence_count > self.sequence_gear):
            raise ValueError(
                f"batch ({real_query_tokens} tokens, {sequence_count} sequences) "
                f"exceeds prefill gear ({self.query_gear}, {self.sequence_gear})")
        if not scheduled:
            raise ValueError("prefill staging requires at least one sequence")

        self._tid_np.fill(0)
        self._pos_np.fill(0)
        self._slot_np.fill(0)
        self._bt_np.fill(0)
        self._sample_np.fill(0)
        if self.active_token_mask is not None:
            self._active_np.fill(False)
            self._active_np[:real_query_tokens] = True
        cumulative_query_lengths = []
        kv_lengths = []
        sample_order = []
        splice_order = []
        splice_rows = []
        query_row = 0

        for sequence_row, item in enumerate(scheduled):
            rid = item.request_id
            req = plan.get_request(rid)
            start = req.num_computed_tokens
            blocks = plan.block_tables[rid]
            self._bt_np[sequence_row, :len(blocks)] = blocks
            for offset in range(item.num_tokens_to_compute):
                position = start + offset
                self._tid_np[query_row] = req.all_token_ids[position]
                self._pos_np[query_row] = position
                self._slot_np[query_row] = slot_mapping(
                    blocks, position, self.block_size)
                query_row += 1
            final_row = query_row - 1
            cumulative_query_lengths.append(query_row)
            kv_lengths.append(start + item.num_tokens_to_compute)
            splice_order.append(rid)
            splice_rows.append(final_row)
            if start + item.num_tokens_to_compute >= req.num_prefill_tokens:
                self._sample_np[len(sample_order)] = final_row
                sample_order.append(rid)

        padding = self.query_gear - real_query_tokens
        if padding:
            padding_row = sequence_count
            padding_blocks = (
                padding + self.block_size - 1) // self.block_size
            for block_col in range(padding_blocks):
                self._bt_np[padding_row, block_col] = (
                    self.scratch0 + block_col)
            for offset in range(padding):
                position = offset
                block = self.scratch0 + offset // self.block_size
                self._tid_np[query_row] = 0
                self._pos_np[query_row] = position
                self._slot_np[query_row] = (
                    block * self.block_size + position % self.block_size)
                query_row += 1
            cumulative_query_lengths.append(self.query_gear)
            kv_lengths.append(padding)

        attention_sequence_count = sequence_count + int(bool(padding))

        self.token_ids.copy_(self._tid_host, non_blocking=self._non_blocking)
        self.positions.copy_(self._pos_host, non_blocking=self._non_blocking)
        self.slots.copy_(self._slot_host, non_blocking=self._non_blocking)
        self.sample_rows.copy_(self._sample_host, non_blocking=self._non_blocking)
        if self.active_token_mask is not None:
            self.active_token_mask.copy_(
                self._active_host, non_blocking=self._non_blocking)

        rows, elements = upload_dirty_block_table(
            self.block_table, self._bt_host, self._bt_np, self._bt_shadow,
            self._non_blocking, active_rows=attention_sequence_count)
        self.copied_block_rows += rows
        self.copied_block_elements += elements

        return StagedPrefillInput(
            self.token_ids, self.positions, self.slots,
            self.block_table[:attention_sequence_count],
            self.sample_rows, cumulative_query_lengths, kv_lengths,
            sample_order, splice_order, splice_rows, self.query_gear,
            real_query_tokens, sequence_count)

    def splice(self, refs, request_order, target_rows) -> None:
        splice_device_tokens(
            self.token_ids, target_rows, request_order, refs)
