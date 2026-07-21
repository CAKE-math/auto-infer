"""Persistent host/device metadata staging for compacted MTP drafting."""
from dataclasses import dataclass

import numpy as np
import torch

from auto_infer.spec_decode.layout import confirmed_layout
from auto_infer.worker.staging import HostStaging, upload_dirty_block_table


@dataclass(frozen=True)
class StagedMtpMetadata:
    block_table: torch.Tensor
    sample_rows: torch.Tensor
    cumulative_query_lengths: list[int]
    kv_lengths: list[int]
    request_count: int
    active_tokens: int
    padding_blocks: tuple[int, ...]


class MtpPipelineStager:
    """Own fixed buffers while exposing exact active drafter metadata views."""

    def __init__(self, *, block_table, sample_rows, block_size: int,
                 scratch0: int, token_capacity: int, geometry):
        self.block_table = block_table
        self.sample_rows = sample_rows
        self.block_size = block_size
        self.scratch0 = scratch0
        self.token_capacity = token_capacity
        self.geometry = geometry
        self.sequence_capacity = block_table.shape[0]
        self.request_capacity = sample_rows.shape[0]
        self.max_blocks = block_table.shape[1]
        host = HostStaging(block_table.device.type != "cpu")
        self._bt_host, self._bt_np = host.allocate(
            tuple(block_table.shape), torch.int32)
        self._sample_host, self._sample_np = host.allocate(
            tuple(sample_rows.shape), torch.long)
        self._non_blocking = host.non_blocking
        self._bt_shadow = np.full(
            tuple(block_table.shape), np.iinfo(np.int32).min,
            dtype=np.int32)
        self.copied_block_rows = 0
        self.copied_block_elements = 0

    def stage_drafter(self, plan, request_order, accepted, *,
                      token_gear: int, request_gear: int) -> StagedMtpMetadata:
        request_order = tuple(request_order)
        accepted = tuple(accepted)
        request_count = len(request_order)
        if len(accepted) != request_count:
            raise ValueError("acceptance flags must match request order")
        if request_count > request_gear or request_gear > self.request_capacity:
            raise ValueError("request count exceeds request gear")
        if token_gear > self.token_capacity:
            raise ValueError("token gear exceeds stager capacity")
        layout = confirmed_layout(accepted, self.geometry)
        if layout.active_tokens > token_gear:
            raise ValueError("confirmed rows exceed token gear")

        padding = token_gear - layout.active_tokens
        attention_sequences = request_count + int(bool(padding))
        if attention_sequences > self.sequence_capacity:
            raise ValueError("drafter sequences exceed stager capacity")

        self._bt_np.fill(0)
        self._sample_np.fill(0)
        kv_lengths = []
        for row, (rid, query_length) in enumerate(
                zip(request_order, layout.query_lengths)):
            blocks = plan.block_tables[rid]
            if len(blocks) > self.max_blocks:
                raise ValueError("request block table exceeds stager capacity")
            self._bt_np[row, :len(blocks)] = blocks
            base = plan.get_request(rid).num_computed_tokens
            kv_lengths.append(base + query_length)
            self._sample_np[row] = layout.final_rows[row]

        cumulative = list(layout.cumulative_query_lengths)
        padding_blocks = ()
        if padding:
            padding_row = request_count
            count = (padding + self.block_size - 1) // self.block_size
            if count > self.max_blocks:
                raise ValueError("padding exceeds scratch block-table capacity")
            padding_blocks = tuple(self.scratch0 + col for col in range(count))
            self._bt_np[padding_row, :count] = padding_blocks
            cumulative.append(token_gear)
            kv_lengths.append(padding)

        self.sample_rows.copy_(
            self._sample_host, non_blocking=self._non_blocking)
        rows, elements = upload_dirty_block_table(
            self.block_table, self._bt_host, self._bt_np, self._bt_shadow,
            self._non_blocking, active_rows=attention_sequences)
        self.copied_block_rows += rows
        self.copied_block_elements += elements

        return StagedMtpMetadata(
            self.block_table[:attention_sequences], self.sample_rows,
            cumulative, kv_lengths, request_count, layout.active_tokens,
            padding_blocks)
