import asyncio

import httpx
import pytest

from auto_infer.engine.executor import Executor
from auto_infer.serving.api_server import create_app
from auto_infer.serving.async_engine import AsyncEngine
from tests.test_serving_load import _cfg
from tests.test_text_serving_api import FakeAsyncEngine, _runtime


def test_shutdown_cancels_active_generation_after_grace():
    class HangingEngine(FakeAsyncEngine):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.aborted = False

        async def generate(self, ids, sampling):
            self.started.set()
            try:
                await asyncio.Event().wait()
                yield ()
            finally:
                self.aborted = True

    async def scenario():
        engine = HangingEngine()
        runtime = await _runtime(engine=engine)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test",
        )
        request = asyncio.create_task(client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 10}
        ))
        await asyncio.wait_for(engine.started.wait(), timeout=1)

        await runtime.shutdown(grace_s=0)

        assert engine.aborted
        assert engine.closed
        assert runtime.admission.snapshot().permits_in_use == 0
        assert not runtime.ready
        assert request.done()
        await asyncio.gather(request, return_exceptions=True)
        await client.aclose()

    asyncio.run(scenario())


def test_runtime_shutdown_is_idempotent():
    async def scenario():
        runtime = await _runtime()

        await runtime.shutdown(grace_s=0)
        await runtime.shutdown(grace_s=0)

        assert runtime.engine.closed
        assert runtime.admission.snapshot().permits_in_use == 0

    asyncio.run(scenario())


def test_invalid_shutdown_grace_does_not_poison_runtime():
    async def scenario():
        runtime = await _runtime()

        with pytest.raises(ValueError, match="grace_s"):
            await runtime.shutdown(grace_s=-1)

        assert runtime.ready
        await runtime.shutdown(grace_s=0)

    asyncio.run(scenario())


def test_fatal_engine_failure_marks_health_unavailable():
    class FatalExecutor(Executor):
        def submit(self, plan, prev_sampled=None):
            raise RuntimeError("fatal device failure")

    async def scenario():
        engine = AsyncEngine(_cfg(), FatalExecutor())
        runtime = await _runtime(engine=engine)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test",
        )

        failed = await client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 1}
        )
        health = await client.get("/health")
        rejected = await client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 1}
        )

        assert failed.status_code == 503
        assert health.status_code == 503
        assert rejected.status_code == 503
        await runtime.shutdown(grace_s=0)
        await client.aclose()

    asyncio.run(scenario())
