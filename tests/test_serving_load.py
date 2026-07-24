import asyncio
import time

import httpx

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.request import SamplingParams
from auto_infer.serving.api_server import (_CountedEvent, _coalesce, create_app)
from auto_infer.serving.async_engine import AsyncEngine
from tests.test_text_serving_api import FakeAsyncEngine, _runtime


def _cfg():
    return EngineConfig(
        model=ModelConfig("/mock"),
        cache=CacheConfig(block_size=4, num_blocks=1000),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
    )


def test_slow_consumer_has_one_pending_slot_and_does_not_block_fast_request():
    async def collect(engine, ids, count):
        output = []
        async for batch in engine.generate(ids, SamplingParams(max_tokens=count)):
            output.extend(batch)
        return output

    async def scenario():
        engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))
        slow = engine.generate([1], SamplingParams(max_tokens=100))
        first = await anext(slow)
        assert first
        await asyncio.sleep(0.02)

        assert engine.pending_output_slots <= 1
        assert await collect(engine, [10], 3) == [11, 12, 13]

        await slow.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_async_engine_delegates_prefix_cache_snapshot():
    engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))
    engine.service._prefix_cache_snapshot = (8, 3)

    assert engine.prefix_cache_snapshot == (8, 3)
    asyncio.run(engine.aclose())


def test_sse_coalescer_flushes_on_deadline_without_waiting_for_next_token():
    async def events():
        yield _CountedEvent("a", 1, 1)
        yield _CountedEvent("b", 2, 1)
        await asyncio.sleep(0.1)
        yield _CountedEvent("", 2, 1, "length")

    async def scenario():
        runtime = await _runtime(sse_coalesce_ms=5)
        stream = _coalesce(events(), runtime.serving_config)

        first = await anext(stream)
        before = time.monotonic()
        second = await asyncio.wait_for(anext(stream), timeout=0.03)

        assert first.text == "a"
        assert second.text == "b"
        assert time.monotonic() - before < 0.03
        await stream.aclose()
        await runtime.shutdown(grace_s=0)

    asyncio.run(scenario())


def test_http_overload_returns_429_without_waiting():
    class HangingEngine(FakeAsyncEngine):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def generate(self, ids, sampling):
            self.started.set()
            await asyncio.Event().wait()
            yield ()

    async def scenario():
        engine = HangingEngine()
        runtime = await _runtime(engine=engine, max_http_inflight=1)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test",
        )
        held = asyncio.create_task(client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 10}
        ))
        await asyncio.wait_for(engine.started.wait(), timeout=1)

        rejected = await client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 1}
        )

        assert rejected.status_code == 429
        assert rejected.json()["error"]["code"] == "queue_full"
        await runtime.shutdown(grace_s=0)
        await asyncio.gather(held, return_exceptions=True)
        await client.aclose()

    asyncio.run(scenario())
