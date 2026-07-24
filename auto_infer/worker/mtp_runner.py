from dataclasses import dataclass

import torch

from auto_infer.engine.token_layout import slot_mapping
from auto_infer.layers.mtp import RecurrentMtpHead
from auto_infer.layers.sampler import stable_greedy


@dataclass(frozen=True)
class MtpItem:
    request_id: str
    hidden: torch.Tensor
    next_tokens: list[int]
    positions: list[int]
    block_table: list[int] | tuple[int, ...]
    generate_drafts: bool = True


class MtpDrafter:
    """Batched recurrent MTP proposer with its own paged KV cache."""

    def __init__(self, head, *, device, block_size):
        self.device = device
        self.block_size = block_size
        self.head = head

    @classmethod
    def from_model(cls, model, num_blocks, block_size, geometry):
        from auto_infer.layers.attention.registry import (
            build_mtp_attention_backend)
        prefix = geometry.layer_prefix(0)
        backend, caches = build_mtp_attention_backend(
            model, "paged", prefix, num_blocks, block_size)
        mask = torch.triu(torch.ones(
            2048, 2048, dtype=torch.int8, device=model.device), diagonal=1)
        head = RecurrentMtpHead(model, backend, caches, mask, prefix)
        return cls(head, device=model.device, block_size=block_size)

    def _forward(self, items):
        hidden = torch.cat([item.hidden for item in items])
        tokens = torch.tensor(
            [token for item in items for token in item.next_tokens],
            dtype=torch.long, device=self.device)
        positions = torch.tensor(
            [position for item in items for position in item.positions],
            dtype=torch.long, device=self.device)
        cumulative, kv_lengths, slots = [], [], []
        offset = 0
        max_blocks = max(len(item.block_table) for item in items)
        block_table = torch.zeros(
            len(items), max_blocks, dtype=torch.int32, device=self.device)
        for row, item in enumerate(items):
            offset += len(item.positions)
            cumulative.append(offset)
            kv_lengths.append(item.positions[-1] + 1)
            slots.extend(slot_mapping(
                item.block_table, position, self.block_size)
                for position in item.positions)
            block_table[row, :len(item.block_table)] = torch.tensor(
                item.block_table, dtype=torch.int32, device=self.device)
        return self.head.forward(
            hidden, tokens, positions,
            torch.tensor(slots, dtype=torch.int32, device=self.device),
            block_table, cumulative, kv_lengths)

    @torch.no_grad()
    def draft(self, items, depth):
        if not items:
            return {}
        hidden, logits = self._forward(items)
        active, drafts = [], {}
        offset = 0
        for item in items:
            offset += len(item.positions)
            if item.generate_drafts:
                row = slice(offset - 1, offset)
                token = int(stable_greedy(
                    hidden[row], logits[row],
                    self.head.model.w["lm_head.weight"])[0])
                drafts[item.request_id] = [token]
                active.append((item, hidden[offset - 1:offset], token))
        for step in range(1, depth):
            if not active:
                break
            continuation = [MtpItem(
                item.request_id, state, [token],
                [item.positions[-1] + step], item.block_table)
                for item, state, token in active]
            hidden, logits = self._forward(continuation)
            next_active = []
            for row, (item, _, _) in enumerate(active):
                token = int(stable_greedy(
                    hidden[row:row + 1], logits[row:row + 1],
                    self.head.model.w["lm_head.weight"])[0])
                drafts[item.request_id].append(token)
                next_active.append((item, hidden[row:row + 1], token))
            active = next_active
        return {rid: tuple(tokens) for rid, tokens in drafts.items()}
