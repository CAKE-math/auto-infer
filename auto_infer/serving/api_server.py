"""Native asynchronous HTTP and SSE frontend for text generation."""

import asyncio
import contextlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from auto_infer.serving.admission import (AdmissionController, Overloaded,
                                          Unavailable)
from auto_infer.serving.config import ServingConfig
from auto_infer.serving.detokenizer import IncrementalTextDecoder, TextDelta
from auto_infer.serving.metrics import ServingMetrics
from auto_infer.serving.protocol import (ChatCompletionRequest,
                                         CompletionRequest, error_payload,
                                         sampling_params, stop_strings)
from auto_infer.serving.tokenizer import (AsyncTokenizer, TokenizerClosed,
                                          TokenizerOverloaded)
from auto_infer.serving.service import EngineQueueFull


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, error_type: str,
                 *, code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.code = code
        self.param = param


class _ArrivalTimestampMiddleware:
    """Pure ASGI timestamping; avoids BaseHTTPMiddleware's streaming tasks."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["received_at"] = time.monotonic()
        await self.app(scope, receive, send)


@dataclass
class ApiRuntime:
    tokenizer: object
    engine: object
    model: str
    max_model_len: int
    serving_config: ServingConfig
    admission: AdmissionController = field(init=False)
    metrics: ServingMetrics = field(init=False)
    async_tokenizer: AsyncTokenizer | None = field(init=False, default=None)
    ready: bool = field(init=False, default=False)
    _shutdown: bool = field(init=False, default=False)
    _active_tasks: set[asyncio.Task] = field(init=False, default_factory=set)
    _shutdown_done: asyncio.Event | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        cfg = self.serving_config
        self.admission = AdmissionController(
            max_http=cfg.max_http_inflight,
            max_engine_requests=cfg.max_waiting_requests,
            max_engine_tokens=cfg.max_waiting_tokens,
        )
        self.metrics = ServingMetrics()

    @property
    def healthy(self) -> bool:
        return self.ready and bool(getattr(self.engine, "healthy", True))

    async def start(self) -> None:
        if self.ready:
            return
        if self._shutdown:
            raise RuntimeError("runtime is shut down")
        cfg = self.serving_config
        self.async_tokenizer = AsyncTokenizer(
            self.tokenizer,
            max_batch_size=cfg.tokenizer_batch_size,
            wait_s=cfg.tokenizer_wait_ms / 1000.0,
            queue_capacity=cfg.tokenizer_queue_capacity,
        )
        self.ready = True

    async def shutdown(self, grace_s: float | None = None) -> None:
        grace = self.serving_config.shutdown_grace_s if grace_s is None else grace_s
        if grace < 0:
            raise ValueError("grace_s must be >= 0")
        if self._shutdown:
            if self._shutdown_done is not None:
                await self._shutdown_done.wait()
            return
        self._shutdown = True
        self._shutdown_done = asyncio.Event()
        self.ready = False
        self.admission.close()
        deadline = time.monotonic() + grace
        while self._active_tasks and time.monotonic() < deadline:
            await asyncio.sleep(min(0.01, deadline - time.monotonic()))
        active = [
            task for task in self._active_tasks
            if task is not asyncio.current_task()
        ]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        try:
            if self.async_tokenizer is not None:
                await self.async_tokenizer.aclose()
            await self.engine.aclose()
        finally:
            self._shutdown_done.set()


def _api_error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status,
        content=error_payload(
            str(error), error_type=error.error_type,
            code=error.code, param=error.param,
        ),
    )


def create_app(runtime: ApiRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(_ArrivalTimestampMiddleware)

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, error: ApiError):
        return _api_error_response(error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, error: RequestValidationError):
        details = error.errors()
        param = ".".join(str(item) for item in details[0]["loc"][1:]) if details else None
        return _api_error_response(ApiError(
            400, "invalid request", "invalid_request_error",
            code="validation_error", param=param,
        ))

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        api_key = runtime.serving_config.api_key
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise ApiError(
                401, "invalid API key", "authentication_error",
                code="invalid_api_key",
            )

    @app.post("/v1/completions", dependencies=[Depends(authorize)])
    async def completions(body: CompletionRequest, request: Request):
        return await _complete(
            runtime, body, chat=False,
            received_at=request.state.received_at,
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        return await _complete(
            runtime, body, chat=True,
            received_at=request.state.received_at,
        )

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": runtime.model,
                "object": "model",
                "created": 0,
                "owned_by": "auto-infer",
            }],
        }

    @app.get("/health")
    async def health():
        if not runtime.healthy:
            raise ApiError(503, "engine unavailable", "service_unavailable")
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        running, waiting, kv_utilization = getattr(
            runtime.engine, "load_snapshot", (0, 0, 0.0)
        )
        runtime.metrics.set_load(
            running=running, waiting=waiting,
            kv_utilization=kv_utilization,
        )
        queried_blocks, hit_blocks = getattr(
            runtime.engine, "prefix_cache_snapshot", (0, 0)
        )
        runtime.metrics.set_prefix_cache(
            queried_blocks=queried_blocks, hit_blocks=hit_blocks
        )
        return Response(
            runtime.metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    return app


async def _complete(runtime: ApiRuntime, request, *, chat: bool,
                    received_at: float | None = None):
    started = time.monotonic()
    runtime.metrics.observe_stage(
        "http_parse", max(0.0, started - (received_at or started))
    )
    admission_started = time.monotonic()
    try:
        http_lease = runtime.admission.acquire_http()
    except Overloaded as error:
        runtime.metrics.record_rejection("http")
        raise ApiError(429, str(error), "overloaded", code="queue_full") from error
    except Unavailable as error:
        raise ApiError(503, str(error), "service_unavailable") from error
    finally:
        runtime.metrics.observe_stage(
            "admission_wait", time.monotonic() - admission_started
        )
    try:
        events = _tracked_events(
            runtime, _generation_events(runtime, request, chat=chat)
        )
        if request.stream:
            return StreamingResponse(
                _stream_response(
                    runtime, request, events, http_lease=http_lease,
                    chat=chat, started=started,
                ),
                media_type="text/event-stream",
            )
        output = []
        final = None
        async for event in events:
            output.append(event.text)
            if event.finished:
                final = event
        if final is None:
            raise ApiError(500, "generation ended without a terminal event",
                           "engine_error")
        runtime.metrics.observe_stage("e2e", time.monotonic() - started)
        runtime.metrics.record_request("chat" if chat else "completion", "ok")
        runtime.metrics.record_tokens(
            prompt=final.prompt_tokens, generated=final.completion_tokens
        )
        return _nonstream_payload(
            runtime, "".join(output), final, chat=chat,
            prompt_tokens=getattr(final, "prompt_tokens", None),
        )
    finally:
        # StreamingResponse consumes the generator after this function returns,
        # so its lease is transferred into the response generator.
        if not request.stream:
            http_lease.release()


@dataclass(frozen=True)
class _CountedEvent:
    text: str
    completion_tokens: int
    prompt_tokens: int
    finish_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


async def _tracked_events(runtime: ApiRuntime,
                          events: AsyncIterator[_CountedEvent]
                          ) -> AsyncIterator[_CountedEvent]:
    task = asyncio.current_task()
    if task is not None:
        runtime._active_tasks.add(task)
    try:
        async for event in events:
            yield event
    finally:
        if task is not None:
            runtime._active_tasks.discard(task)
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


async def _tokenize(runtime: ApiRuntime, request, *, chat: bool) -> list[int]:
    tokenizer = runtime.async_tokenizer
    if tokenizer is None:
        raise ApiError(503, "tokenizer unavailable", "service_unavailable")
    before = time.monotonic()
    try:
        if chat:
            ids = await tokenizer.render_chat(
                [message.model_dump() for message in request.messages],
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            ids = await tokenizer.encode(request.prompt)
    except TokenizerOverloaded as error:
        runtime.metrics.record_rejection("tokenizer")
        raise ApiError(429, str(error), "overloaded", code="queue_full") from error
    except TokenizerClosed as error:
        raise ApiError(503, str(error), "service_unavailable") from error
    finally:
        runtime.metrics.observe_stage("tokenize", time.monotonic() - before)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    while ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    token_ids = [int(token) for token in ids]
    if not token_ids:
        raise ApiError(400, "prompt must produce at least one token",
                       "invalid_request_error", param="prompt")
    return token_ids


async def _generation_events(runtime: ApiRuntime, request, *, chat: bool
                             ) -> AsyncIterator[_CountedEvent]:
    if not runtime.healthy:
        runtime.admission.close()
        raise ApiError(503, "engine unavailable", "service_unavailable")
    ids = await _tokenize(runtime, request, chat=chat)
    if len(ids) + request.max_tokens > runtime.max_model_len:
        raise ApiError(
            400, "prompt and max_tokens exceed max_model_len",
            "invalid_request_error", code="context_length_exceeded",
        )
    params = sampling_params(request, runtime.tokenizer.eos_token_id)
    max_kv_tokens = getattr(runtime.engine, "max_kv_tokens", None)
    required_kv_tokens = len(ids) + max(0, params.max_tokens - 1)
    if max_kv_tokens is not None and required_kv_tokens > max_kv_tokens:
        raise ApiError(
            400, "request exceeds available KV cache capacity",
            "invalid_request_error", code="kv_capacity_exceeded",
        )
    try:
        engine_lease = runtime.admission.acquire_engine(prompt_tokens=len(ids))
    except Overloaded as error:
        runtime.metrics.record_rejection("engine")
        raise ApiError(429, str(error), "overloaded", code="queue_full") from error
    except Unavailable as error:
        raise ApiError(503, str(error), "service_unavailable") from error

    decoder = IncrementalTextDecoder(
        runtime.tokenizer, stop_strings(request)
    )
    completion_tokens = 0
    first_token_at = None
    previous_token_at = None
    generation_started = time.monotonic()
    stream = runtime.engine.generate(ids, params)
    engine_stage_recorded = False
    try:
        async for batch in stream:
            for decode_seconds in getattr(batch, "decode_seconds", ()):
                runtime.metrics.observe_stage("decode", decode_seconds)
            if not engine_stage_recorded:
                queue_seconds = getattr(batch, "engine_queue_seconds", None)
                prefill_seconds = getattr(batch, "prefill_seconds", None)
                if queue_seconds is not None:
                    runtime.metrics.observe_stage("engine_queue", queue_seconds)
                if prefill_seconds is not None:
                    runtime.metrics.observe_stage("prefill", prefill_seconds)
                engine_stage_recorded = True
            for token in batch:
                now = time.monotonic()
                next_count = completion_tokens + 1
                is_stop = next_count >= params.min_tokens and (
                    token in params.stop_token_ids or (
                        not params.ignore_eos
                        and params.eos_token_id is not None
                        and token == params.eos_token_id
                    )
                )
                if is_stop:
                    delta = decoder.finish("stop")
                    yield _event(delta, len(ids))
                    return
                before_decode = time.monotonic()
                delta = decoder.push((int(token),))
                runtime.metrics.observe_stage(
                    "detokenize", time.monotonic() - before_decode
                )
                completion_tokens += 1
                if first_token_at is None:
                    first_token_at = now
                    runtime.metrics.observe_stage("ttft", now - generation_started)
                elif previous_token_at is not None:
                    runtime.metrics.observe_stage("itl", now - previous_token_at)
                previous_token_at = now
                yield _event(delta, len(ids))
                if delta.finished:
                    return
                if completion_tokens >= params.max_tokens:
                    yield _event(decoder.finish("length"), len(ids))
                    return
        yield _event(decoder.finish("length"), len(ids))
    except asyncio.CancelledError:
        runtime.metrics.record_abort()
        raise
    except EngineQueueFull as error:
        runtime.metrics.record_rejection("engine_queue")
        raise ApiError(
            429, str(error), "overloaded", code="queue_full"
        ) from error
    except ApiError:
        raise
    except BaseException as error:
        if not bool(getattr(runtime.engine, "healthy", True)):
            runtime.ready = False
            runtime.admission.close()
            runtime.metrics.record_error("fatal_engine")
            raise ApiError(
                503, str(error), "service_unavailable",
                code="engine_unhealthy",
            ) from error
        runtime.metrics.record_error("engine")
        raise ApiError(500, str(error), "engine_error") from error
    finally:
        engine_lease.release()
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


def _event(delta: TextDelta, prompt_tokens: int) -> _CountedEvent:
    return _CountedEvent(
        text=delta.text,
        completion_tokens=delta.token_count,
        prompt_tokens=prompt_tokens,
        finish_reason=delta.finish_reason,
    )


def _nonstream_payload(runtime: ApiRuntime, text: str, final: _CountedEvent,
                       *, chat: bool, prompt_tokens: int | None):
    request_id = _request_id(chat)
    choice = ({
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": final.finish_reason,
    } if chat else {
        "index": 0,
        "text": text,
        "finish_reason": final.finish_reason,
    })
    prompt_count = final.prompt_tokens if prompt_tokens is None else prompt_tokens
    return {
        "id": request_id,
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": runtime.model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_count,
            "completion_tokens": final.completion_tokens,
            "total_tokens": prompt_count + final.completion_tokens,
        },
    }


async def _coalesce(events: AsyncIterator[_CountedEvent], config: ServingConfig
                    ) -> AsyncIterator[_CountedEvent]:
    iterator = events.__aiter__()
    pending: asyncio.Task | None = None
    text = ""
    first_sent = False
    buffered_tokens = 0
    last_count = 0
    deadline: float | None = None
    latest: _CountedEvent | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            if deadline is None:
                done = None
            else:
                timeout = max(0.0, deadline - time.monotonic())
                done, _ = await asyncio.wait({pending}, timeout=timeout)
                if not done:
                    assert latest is not None
                    yield _CountedEvent(
                        text=text,
                        completion_tokens=latest.completion_tokens,
                        prompt_tokens=latest.prompt_tokens,
                    )
                    text = ""
                    buffered_tokens = 0
                    deadline = None
                    continue
            try:
                event = await pending if done is None else pending.result()
            except StopAsyncIteration:
                if text and latest is not None:
                    yield _CountedEvent(
                        text=text,
                        completion_tokens=latest.completion_tokens,
                        prompt_tokens=latest.prompt_tokens,
                        finish_reason=latest.finish_reason,
                    )
                return
            finally:
                if pending.done():
                    pending = None

            latest = event
            new_tokens = max(0, event.completion_tokens - last_count)
            last_count = event.completion_tokens
            if not first_sent and event.text:
                first_sent = True
                yield _CountedEvent(
                    text=text + event.text,
                    completion_tokens=event.completion_tokens,
                    prompt_tokens=event.prompt_tokens,
                    finish_reason=event.finish_reason,
                )
                text = ""
                buffered_tokens = 0
                deadline = None
                continue

            if not text and event.text:
                deadline = time.monotonic() + config.sse_coalesce_ms / 1000.0
            text += event.text
            buffered_tokens += new_tokens
            if (event.finished
                    or buffered_tokens >= config.sse_coalesce_tokens
                    or config.sse_coalesce_ms == 0):
                yield _CountedEvent(
                    text=text,
                    completion_tokens=event.completion_tokens,
                    prompt_tokens=event.prompt_tokens,
                    finish_reason=event.finish_reason,
                )
                text = ""
                buffered_tokens = 0
                deadline = None
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


async def _stream_response(runtime: ApiRuntime, request,
                           events: AsyncIterator[_CountedEvent], *, chat: bool,
                           started: float, http_lease) -> AsyncIterator[str]:
    request_id = _request_id(chat)
    final_event = None
    try:
        async for event in _coalesce(events, runtime.serving_config):
            final_event = event
            choice = ({
                "index": 0,
                "delta": {"content": event.text},
                "finish_reason": event.finish_reason,
            } if chat else {
                "index": 0,
                "text": event.text,
                "finish_reason": event.finish_reason,
            })
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk" if chat else "text_completion.chunk",
                "created": int(time.time()),
                "model": runtime.model,
                "completion_tokens": event.completion_tokens,
                "choices": [choice],
            }
            before_send = time.monotonic()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            runtime.metrics.observe_stage(
                "sse_send", time.monotonic() - before_send
            )
        yield "data: [DONE]\n\n"
        runtime.metrics.observe_stage("e2e", time.monotonic() - started)
        runtime.metrics.record_request("chat" if chat else "completion", "ok")
        if final_event is not None:
            runtime.metrics.record_tokens(
                prompt=final_event.prompt_tokens,
                generated=final_event.completion_tokens,
            )
    except ApiError as error:
        runtime.metrics.record_error(error.error_type)
        yield "data: " + json.dumps(error_payload(
            str(error), error_type=error.error_type,
            code=error.code, param=error.param,
        )) + "\n\n"
    finally:
        http_lease.release()


def _request_id(chat: bool) -> str:
    prefix = "chatcmpl" if chat else "cmpl"
    return f"{prefix}-{uuid.uuid4().hex}"


def build_runtime(*, tokenizer, engine, model: str, max_model_len: int,
                  serving_config: ServingConfig) -> ApiRuntime:
    return ApiRuntime(
        tokenizer=tokenizer,
        engine=engine,
        model=model,
        max_model_len=max_model_len,
        serving_config=serving_config,
    )


def run_runtime(runtime: ApiRuntime, host: str = "0.0.0.0", port: int = 8000,
                access_log: bool = False) -> None:
    import uvicorn

    uvicorn.run(
        create_app(runtime), host=host, port=port, access_log=access_log
    )


def build_engine_config(
    *,
    model_path: str,
    model_package: str | None,
    device_index: int,
    mode: str,
    max_model_len: int,
    num_blocks: int,
    block_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_gear: int,
    max_prefill_tokens: int,
    num_speculative_tokens: int,
    parallel=None,
):
    from auto_infer.config import (
        CacheConfig,
        EngineConfig,
        ExecutionConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpecDecodeConfig,
    )

    return EngineConfig(
        model=ModelConfig(
            model_path=model_path, max_model_len=max_model_len,
            model_package=model_package),
        parallel=parallel or ParallelConfig(),
        cache=CacheConfig(block_size=block_size, num_blocks=num_blocks),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        execution=ExecutionConfig(
            mode=mode, device_index=device_index, max_gear=max_gear,
            max_prefill_tokens=max_prefill_tokens,
        ),
        spec_decode=(SpecDecodeConfig(num_speculative_tokens)
                     if mode == "graph_mtp" else None),
    )


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8000,
          model_package: str | None = None,
          device_index: int = 0, mode: str = "paged", max_model_len: int = 4096,
          num_blocks: int = 4096, block_size: int = 16, max_num_seqs: int = 256,
          max_num_batched_tokens: int = 8192, max_gear: int = 32,
          max_prefill_tokens: int = 256,
          num_speculative_tokens: int = 1,
          access_log: bool = False,
          serving_config: ServingConfig | None = None) -> None:
    from transformers import AutoTokenizer

    from auto_infer.engine.factory import build_executor
    from auto_infer.serving.async_engine import AsyncEngine

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    engine_config = build_engine_config(
        model_path=model_path,
        model_package=model_package,
        device_index=device_index,
        mode=mode,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_gear=max_gear,
        max_prefill_tokens=max_prefill_tokens,
        num_speculative_tokens=num_speculative_tokens,
    )
    resolved_serving_config = serving_config or ServingConfig(
        max_num_seqs=max_num_seqs
    )
    engine = AsyncEngine(
        engine_config, build_executor(engine_config),
        inbox_capacity=resolved_serving_config.max_waiting_requests,
    )
    runtime = build_runtime(
        tokenizer=tokenizer,
        engine=engine,
        model=model_path.rstrip("/").split("/")[-1],
        max_model_len=max_model_len,
        serving_config=resolved_serving_config,
    )
    run_runtime(
        runtime, host=host, port=port, access_log=access_log
    )
