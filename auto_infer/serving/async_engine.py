import asyncio
from dataclasses import dataclass

from auto_infer.engine.request import SamplingParams
from auto_infer.serving.broker import AsyncOutputCollector
from auto_infer.serving.service import EngineService


@dataclass(frozen=True)
class EngineTokenBatch:
    tokens: tuple[int, ...]
    engine_queue_seconds: float | None = None
    prefill_seconds: float | None = None
    decode_seconds: tuple[float, ...] = ()

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __bool__(self):
        return bool(self.tokens)


class AsyncEngine:
    """Async facade using the same persistent EngineService and broker."""

    def __init__(self, config, executor, *, inbox_capacity: int | None = None):
        self.service = EngineService(
            config, executor, inbox_capacity=inbox_capacity
        )
        self.engine = self.service.engine
        self._active: dict[str, AsyncOutputCollector] = {}
        self._closed = False

    @classmethod
    def from_service(cls, service):
        instance = cls.__new__(cls)
        instance.service = service
        instance.engine = service.engine
        instance._active = {}
        instance._closed = False
        return instance

    @property
    def pending_output_slots(self) -> int:
        return max(
            (collector.pending_slots for collector in self._active.values()),
            default=0,
        )

    @property
    def healthy(self) -> bool:
        return not self._closed and self.service.healthy

    @property
    def max_kv_tokens(self) -> int:
        return self.service.max_kv_tokens

    @property
    def load_snapshot(self) -> tuple[int, int, float]:
        return self.service.load_snapshot

    @property
    def prefix_cache_snapshot(self) -> tuple[int, int]:
        return self.service.prefix_cache_snapshot

    async def generate(self, ids, sampling: SamplingParams | None = None):
        if self._closed:
            raise RuntimeError("async engine is closed")
        params = sampling or SamplingParams()
        collector = AsyncOutputCollector(asyncio.get_running_loop())
        rid = self.service.submit_to(ids, params, collector)
        self._active[rid] = collector
        first = True
        try:
            while True:
                tokens = await collector.get()
                if tokens is None:
                    return
                timing = (self.service.take_first_stage_timing(rid)
                          if first else None)
                decode_timings = self.service.take_decode_stage_timings(rid)
                first = False
                yield EngineTokenBatch(
                    tokens,
                    engine_queue_seconds=timing[0] if timing else None,
                    prefill_seconds=timing[1] if timing else None,
                    decode_seconds=decode_timings,
                )
        finally:
            self._active.pop(rid, None)
            self.service.release(rid)

    def abort(self, request_id: str) -> None:
        collector = self._active.pop(request_id, None)
        if collector is not None:
            collector.finish()
        self.service.release(request_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rid, collector in list(self._active.items()):
            collector.finish()
            self.service.release(rid)
        self._active.clear()
        self.service.close()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rid, collector in list(self._active.items()):
            collector.finish()
            self.service.release(rid)
        self._active.clear()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.service.close)
