import asyncio

import pytest

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.request import SamplingParams
from auto_infer.serving.async_engine import AsyncEngine
from auto_infer.serving.broker import AsyncOutputCollector


def _cfg():
    return EngineConfig(
        model=ModelConfig("/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
    )


def test_collector_merges_when_producer_gets_ahead():
    async def scenario():
        collector = AsyncOutputCollector(asyncio.get_running_loop())

        collector.put_tokens((1, 2))
        collector.put_tokens((3, 4))

        assert await collector.get() == (1, 2, 3, 4)
        assert collector.pending_slots == 0

    asyncio.run(scenario())


def test_collector_delivers_pending_tokens_before_completion():
    async def scenario():
        collector = AsyncOutputCollector(asyncio.get_running_loop())
        collector.put_tokens((7, 8))
        collector.finish()

        assert await collector.get() == (7, 8)
        assert await collector.get() is None

    asyncio.run(scenario())


def test_collector_failure_wakes_consumer():
    async def scenario():
        collector = AsyncOutputCollector(asyncio.get_running_loop())
        waiter = asyncio.create_task(collector.get())
        collector.fail(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await waiter

    asyncio.run(scenario())


def test_async_generate_uses_no_to_thread(monkeypatch):
    async def scenario():
        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *args, **kwargs: pytest.fail("blocking bridge used"),
        )
        engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))

        batches = [
            batch
            async for batch in engine.generate(
                [1, 2, 3], SamplingParams(max_tokens=3)
            )
        ]

        assert [token for batch in batches for token in batch] == [4, 5, 6]
        await engine.aclose()

    asyncio.run(scenario())


def test_async_generate_exposes_first_token_stage_timings():
    async def scenario():
        engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))

        first = await anext(engine.generate(
            [1, 2, 3], SamplingParams(max_tokens=1)
        ))

        assert tuple(first) == (4,)
        assert first.engine_queue_seconds >= 0
        assert first.prefill_seconds >= 0
        await engine.aclose()

    asyncio.run(scenario())


def test_async_generate_preserves_engine_decode_timings_when_outputs_merge():
    async def scenario():
        engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))
        batches = [
            batch async for batch in engine.generate(
                [1, 2, 3], SamplingParams(max_tokens=4)
            )
        ]

        assert sum(len(batch.decode_seconds) for batch in batches) == 3
        assert all(
            value >= 0 for batch in batches for value in batch.decode_seconds
        )
        await engine.aclose()

    asyncio.run(scenario())


def test_collector_schedules_only_one_callback_while_loop_is_stalled():
    class StalledLoop:
        def __init__(self):
            self.callbacks = []

        def call_soon_threadsafe(self, callback, *args):
            self.callbacks.append((callback, args))

    loop = StalledLoop()
    collector = AsyncOutputCollector(loop)

    for token in range(100):
        collector.put_tokens((token,))

    assert len(loop.callbacks) == 1
    assert collector.pending_slots == 1
