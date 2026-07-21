import asyncio

import pytest

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.request import SamplingParams
from auto_infer.serving.async_engine import AsyncEngine
from auto_infer.serving.service import EngineService


def _cfg():
    return EngineConfig(
        model=ModelConfig("/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64))


def test_service_close_joins_thread_and_rejects_submissions():
    service = EngineService(_cfg(), MockExecutor(vocab_size=1000))
    service.close()
    assert not service.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        service.submit([1], SamplingParams(max_tokens=1))


def test_cancelled_async_request_does_not_kill_service():
    async def scenario():
        engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))
        async def consume(ids, max_tokens):
            output = []
            async for batch in engine.generate(
                    ids, SamplingParams(max_tokens=max_tokens)):
                output.extend(batch)
            return output

        task = asyncio.create_task(consume([1, 2, 3], max_tokens=100))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await consume([10], max_tokens=2) == [11, 12]
        await engine.aclose()
        assert not engine.service.thread.is_alive()

    asyncio.run(scenario())
