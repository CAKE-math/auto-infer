import asyncio
import json

import httpx

from auto_infer.serving.api_server import ApiRuntime, create_app
from auto_infer.serving.async_engine import EngineTokenBatch
from auto_infer.serving.config import ServingConfig
from auto_infer.serving.service import EngineQueueFull


class TextTokenizer:
    eos_token_id = 99
    bos_token_id = None

    def __call__(self, prompts, **kwargs):
        return {"input_ids": [[1, 2, 3] for _ in prompts]}

    def batch_decode(self, token_ids, **kwargs):
        return ["".join(chr(96 + token) for token in ids) for ids in token_ids]

    def apply_chat_template(self, messages, **kwargs):
        return [10, 11]

    def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
        return [chr(96 + token) for token in token_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class FakeAsyncEngine:
    def __init__(self):
        self.closed = False

    async def generate(self, ids, sampling):
        for index, token in enumerate((4, 5, 6)):
            await asyncio.sleep(0)
            yield EngineTokenBatch(
                (token,), decode_seconds=(0.001,) if index else ()
            )

    async def aclose(self):
        self.closed = True

    @property
    def load_snapshot(self):
        return (2, 3, 0.25)

    @property
    def prefix_cache_snapshot(self):
        return (8, 3)


async def _runtime(*, api_key=None, engine=None, max_model_len=128,
                   max_http_inflight=8, sse_coalesce_ms=0):
    runtime = ApiRuntime(
        tokenizer=TextTokenizer(),
        engine=engine or FakeAsyncEngine(),
        model="mock-model",
        max_model_len=max_model_len,
        serving_config=ServingConfig(
            max_num_seqs=4,
            max_http_inflight=max_http_inflight,
            max_waiting_requests=8,
            max_waiting_tokens=1024,
            tokenizer_wait_ms=0,
            sse_coalesce_ms=sse_coalesce_ms,
            api_key=api_key,
        ),
    )
    await runtime.start()
    return runtime


def _sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def test_streaming_and_nonstreaming_completion_match():
    async def scenario():
        runtime = await _runtime()
        app = create_app(runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            plain = await client.post(
                "/v1/completions", json={"prompt": "abc", "max_tokens": 3}
            )
            streamed = await client.post(
                "/v1/completions",
                json={"prompt": "abc", "max_tokens": 3, "stream": True},
            )
        assert runtime.admission.snapshot().permits_in_use == 0
        await runtime.shutdown()

        assert plain.status_code == 200
        assert plain.json()["choices"][0] == {
            "index": 0,
            "text": "def",
            "finish_reason": "length",
        }
        assert plain.json()["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 3,
            "total_tokens": 6,
        }
        payloads = _sse_payloads(streamed.text)
        assert "".join(
            item["choices"][0]["text"] for item in payloads[:-1]
        ) == "def"
        assert len({item["id"] for item in payloads[:-1]}) == 1
        assert payloads[-1] == "[DONE]"

    asyncio.run(scenario())


def test_chat_models_health_and_metrics_endpoints():
    async def scenario():
        runtime = await _runtime()
        app = create_app(runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            chat = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 3,
                },
            )
            models = await client.get("/v1/models")
            health = await client.get("/health")
            metrics = await client.get("/metrics")
        await runtime.shutdown()

        assert chat.json()["choices"][0]["message"] == {
            "role": "assistant",
            "content": "def",
        }
        assert models.json()["data"][0]["id"] == "mock-model"
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert "auto_infer_serving_requests_total" in metrics.text
        assert "auto_infer_serving_running_requests 2.0" in metrics.text
        assert "auto_infer_serving_waiting_requests 3.0" in metrics.text
        assert "auto_infer_serving_kv_cache_utilization 0.25" in metrics.text
        assert "auto_infer_serving_prefix_cache_queried_blocks 8.0" in metrics.text
        assert "auto_infer_serving_prefix_cache_hit_blocks 3.0" in metrics.text
        assert "auto_infer_serving_prefix_cache_hit_rate 0.375" in metrics.text
        assert 'auto_infer_serving_tokens_total{kind="prompt"} 2.0' in metrics.text
        assert 'auto_infer_serving_tokens_total{kind="generated"} 3.0' in metrics.text
        assert "auto_infer_serving_process_cpu_seconds" in metrics.text
        assert "auto_infer_serving_process_peak_rss_bytes" in metrics.text
        for stage in ("http_parse", "admission_wait", "tokenize", "decode",
                      "ttft", "e2e"):
            prefix = f"auto_infer_serving_{stage}_seconds_count "
            count = next(
                float(line.removeprefix(prefix))
                for line in metrics.text.splitlines()
                if line.startswith(prefix)
            )
            assert count > 0

    asyncio.run(scenario())


def test_request_larger_than_engine_kv_capacity_is_structured_400():
    class SmallEngine(FakeAsyncEngine):
        max_kv_tokens = 4

    async def scenario():
        runtime = await _runtime(engine=SmallEngine())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "abc", "max_tokens": 3}
            )
        snapshot = runtime.admission.snapshot()
        await runtime.shutdown()

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "kv_capacity_exceeded"
        assert snapshot.engine_requests == 0

    asyncio.run(scenario())


def test_engine_submission_queue_full_is_structured_429():
    class FullEngine(FakeAsyncEngine):
        async def generate(self, ids, sampling):
            raise EngineQueueFull("engine submission queue is full")
            yield ()

    async def scenario():
        runtime = await _runtime(engine=FullEngine())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(runtime)),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "abc", "max_tokens": 1}
            )
        snapshot = runtime.admission.snapshot()
        await runtime.shutdown()

        assert response.status_code == 429
        assert response.json()["error"]["code"] == "queue_full"
        assert snapshot.engine_requests == 0

    asyncio.run(scenario())


def test_bearer_auth_and_validation_errors_are_structured():
    async def scenario():
        runtime = await _runtime(api_key="secret")
        app = create_app(runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            unauthorized = await client.post(
                "/v1/completions", json={"prompt": "abc"}
            )
            invalid = await client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": "abc", "max_tokens": 0},
            )
            invalid_limits = await client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": "abc", "max_tokens": 1, "min_tokens": 2},
            )
            accepted = await client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer secret"},
                json={"prompt": "abc", "max_tokens": 1},
            )
        await runtime.shutdown()

        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["type"] == "authentication_error"
        assert invalid.status_code == 400
        assert invalid_limits.status_code == 400
        assert runtime.admission.snapshot().engine_requests == 0
        assert invalid.json()["error"]["type"] == "invalid_request_error"
        assert accepted.status_code == 200

    asyncio.run(scenario())


def test_eos_stop_string_context_and_unavailable_errors():
    class EosEngine(FakeAsyncEngine):
        async def generate(self, ids, sampling):
            yield (4, 99, 5)

    async def scenario():
        runtime = await _runtime(engine=EosEngine())
        app = create_app(runtime)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            eos = await client.post(
                "/v1/completions", json={"prompt": "abc", "max_tokens": 5}
            )
            stopped = await client.post(
                "/v1/completions",
                json={"prompt": "abc", "max_tokens": 5, "stop": "d"},
            )
            runtime.admission.close()
            unavailable = await client.post(
                "/v1/completions", json={"prompt": "abc"}
            )
        await runtime.shutdown()

        assert eos.json()["choices"][0]["text"] == "d"
        assert eos.json()["choices"][0]["finish_reason"] == "stop"
        assert eos.json()["usage"]["completion_tokens"] == 1
        assert stopped.json()["choices"][0]["text"] == ""
        assert stopped.json()["choices"][0]["finish_reason"] == "stop"
        assert unavailable.status_code == 503
        assert unavailable.json()["error"]["type"] == "service_unavailable"

    asyncio.run(scenario())


def test_context_and_engine_failures_are_structured():
    class FailingEngine(FakeAsyncEngine):
        async def generate(self, ids, sampling):
            raise RuntimeError("device failed")
            yield ()

    async def scenario():
        short = await _runtime(max_model_len=4)
        failing = await _runtime(engine=FailingEngine())
        short_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(short)),
            base_url="http://test",
        )
        failing_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(failing)),
            base_url="http://test",
        )
        context = await short_client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 2}
        )
        engine = await failing_client.post(
            "/v1/completions", json={"prompt": "abc", "max_tokens": 2}
        )
        await short_client.aclose()
        await failing_client.aclose()
        await short.shutdown()
        await failing.shutdown()

        assert context.status_code == 400
        assert context.json()["error"]["code"] == "context_length_exceeded"
        assert engine.status_code == 500
        assert engine.json()["error"]["type"] == "engine_error"

    asyncio.run(scenario())
