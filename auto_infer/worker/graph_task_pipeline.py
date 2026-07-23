"""Prepare dynamic ACL graph-task metadata before graph replay submission."""
from dataclasses import replace
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphTaskTicket:
    context: object


class GraphTaskPipeline:
    """Per-gear replay/update coordinator.

    ``stream_context`` is injectable so ordering and metadata ownership remain
    host-testable without importing torch-npu.
    """

    def __init__(self, backend, update_stream, metadata_slots: int = 2,
                 stream_context=None, registrations=None):
        if metadata_slots < 2:
            raise ValueError("metadata_slots must be at least 2")
        if stream_context is None:
            import torch
            stream_context = torch.npu.stream
        self.backend = backend
        self.registrations = registrations
        self.update_stream = update_stream
        self._stream_context = stream_context
        self._kv_slots = [[] for _ in range(metadata_slots)]
        self._query_slots = [[] for _ in range(metadata_slots)]
        self._multi_kv_slots = [[] for _ in range(metadata_slots)]
        self._multi_query_slots = [[] for _ in range(metadata_slots)]
        self._next_slot = 0

    def _stage_context(self, ctx):
        kv_slot = self._kv_slots[self._next_slot]
        query_slot = self._query_slots[self._next_slot]
        kv_slot[:] = ctx.seqlens_kv
        query_slot[:] = ctx.cu_seqlens_q or []
        self._next_slot = (self._next_slot + 1) % len(self._kv_slots)
        return replace(
            ctx, seqlens_kv=kv_slot, cu_seqlens_q=query_slot)

    def prepare(self, ctx) -> GraphTaskTicket:
        staged = self._stage_context(ctx)
        with self._stream_context(self.update_stream):
            if (self.registrations is not None
                    and hasattr(self.backend, "update_registrations")):
                self.backend.update_registrations(
                    self.registrations, staged, stream=self.update_stream)
            else:
                self.backend.update(staged, stream=self.update_stream)
        return GraphTaskTicket(staged)

    @staticmethod
    def submit(graph, ticket: GraphTaskTicket) -> None:
        graph.replay()

    def replay(self, graph, ctx) -> None:
        """Compatibility for synchronous/MTP callers."""
        self.submit(graph, self.prepare(ctx))

    def replay_many(self, graph, contexts) -> None:
        """Replay one graph containing multiple dynamic attention tasks."""
        slot = self._next_slot
        kv_bank = self._multi_kv_slots[slot]
        query_bank = self._multi_query_slots[slot]
        while len(kv_bank) < len(contexts):
            kv_bank.append([])
            query_bank.append([])
        staged = []
        for index, ctx in enumerate(contexts):
            kv_bank[index][:] = ctx.seqlens_kv
            query_bank[index][:] = ctx.cu_seqlens_q or []
            staged.append(replace(
                ctx, seqlens_kv=kv_bank[index],
                cu_seqlens_q=query_bank[index]))
        self._next_slot = (slot + 1) % len(self._multi_kv_slots)
        with self._stream_context(self.update_stream):
            self.backend.update_many(staged, stream=self.update_stream)
        graph.replay()
