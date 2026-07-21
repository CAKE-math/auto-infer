# Native Async Text Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thread-per-request HTTP path with a bounded native-async text-generation Serving layer that covers the approved core vLLM functionality and preserves auto-infer Engine performance.

**Architecture:** FastAPI/uvicorn owns HTTP and SSE coroutines; a bounded async tokenizer and admission controller protect host work. A dedicated EngineService thread remains the sole EngineCore owner and publishes token batches into single-slot asyncio collectors without blocking on clients or sockets.

**Tech Stack:** Python 3.11, asyncio, FastAPI, uvicorn, Pydantic, prometheus-client, pytest, httpx, existing auto-infer EngineCore and MockExecutor.

## Global Constraints

- In-scope endpoints are only `POST /v1/completions`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`, and `GET /metrics`.
- Embeddings, pooling, reranking, Responses API, dynamic LoRA, audio, transcription, Realtime, and multimodal APIs remain out of scope.
- EngineCore has one owner thread; HTTP, tokenization, metrics, and socket writes never execute in the EngineCore hot path.
- No request may create a blocking worker thread or wait on `queue.Queue.get()` through `asyncio.to_thread()`.
- All queues are bounded; saturation returns `429`, unhealthy/shutdown returns `503`.
- First SSE token is immediate; later deltas default to a 5 ms coalescing window.
- Tokenizer defaults are batch size 32 and 2 ms batch wait.
- Waiting-request default is `2 * max_num_seqs`; graceful shutdown defaults to 30 seconds.
- Metrics must not synchronize the NPU.
- Use TDD and keep the worktree clean after each task commit.

---

### Task 1: Serving Configuration and Protocol Boundary

**Files:**
- Modify: `pyproject.toml`
- Create: `auto_infer/serving/config.py`
- Create: `auto_infer/serving/protocol.py`
- Create: `tests/test_serving_protocol.py`

**Interfaces:**
- Produces: `ServingConfig`, `CompletionRequest`, `ChatCompletionRequest`, `OpenAIError`, `sampling_params(request, eos_token_id)`.
- Consumes: `auto_infer.engine.request.SamplingParams`.

- [ ] **Step 1: Write failing protocol and configuration tests**

```python
def test_serving_defaults_and_derived_waiting_limit():
    cfg = ServingConfig(max_num_seqs=8)
    assert cfg.max_waiting_requests == 16
    assert cfg.tokenizer_batch_size == 32
    assert cfg.tokenizer_wait_ms == 2.0
    assert cfg.sse_coalesce_ms == 5.0

def test_completion_rejects_invalid_sampling():
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", max_tokens=0)

def test_protocol_maps_all_supported_sampling_fields():
    req = CompletionRequest(prompt="x", max_tokens=7, temperature=0.3,
                            top_p=0.9, top_k=10, stop_token_ids=[4])
    params = sampling_params(req, eos_token_id=2)
    assert (params.max_tokens, params.temperature, params.top_p,
            params.top_k, params.stop_token_ids) == (7, 0.3, 0.9, 10, [4])
```

- [ ] **Step 2: Run the tests and confirm missing modules fail**

Run: `pytest -q tests/test_serving_protocol.py`

Expected: collection fails because `auto_infer.serving.config` and `protocol` do not exist.

- [ ] **Step 3: Add runtime dependencies and implement typed boundaries**

Add `fastapi`, `uvicorn`, and `prometheus-client` to project dependencies. Implement frozen `ServingConfig` with positive-value validation and a derived waiting limit. Implement Pydantic request models for one text prompt or chat message list, supported sampling fields, `stream`, `stop`, and `stop_token_ids`. Configure models to reject unknown fields. Implement one mapping function to `SamplingParams` and one OpenAI-shaped error builder.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_serving_protocol.py tests/test_request.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml auto_infer/serving/config.py auto_infer/serving/protocol.py tests/test_serving_protocol.py
git commit -m "feat: define text serving protocol"
```

### Task 2: Bounded Async Tokenizer

**Files:**
- Create: `auto_infer/serving/tokenizer.py`
- Create: `tests/test_async_tokenizer.py`

**Interfaces:**
- Produces: `AsyncTokenizer(tokenizer, *, max_batch_size, wait_s, queue_capacity)`, `encode(text, kwargs)`, `decode(token_ids, kwargs)`, `render_chat(messages)`, and `aclose()`.
- Consumes: a Hugging Face tokenizer-like object and `ServingConfig` limits.

- [ ] **Step 1: Write failing micro-batch, bounded-queue, and shutdown tests**

```python
async def test_compatible_encodes_are_microbatched():
    backend = RecordingTokenizer()
    tok = AsyncTokenizer(backend, max_batch_size=4, wait_s=0.01,
                         queue_capacity=8)
    outputs = await asyncio.gather(tok.encode("a"), tok.encode("b"))
    assert outputs == [[1], [2]]
    assert backend.batch_calls == [["a", "b"]]
    await tok.aclose()

async def test_full_tokenizer_queue_rejects_without_blocking():
    tok = AsyncTokenizer(BlockingTokenizer(), max_batch_size=1, wait_s=0,
                         queue_capacity=1)
    first = asyncio.create_task(tok.encode("held"))
    await asyncio.sleep(0)
    with pytest.raises(TokenizerOverloaded):
        await tok.encode("rejected")
    first.cancel()
    await tok.aclose()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_async_tokenizer.py`

Expected: missing `AsyncTokenizer` failure.

- [ ] **Step 3: Implement one executor-backed micro-batcher**

Use one bounded `asyncio.Queue`, one batching task, and one dedicated `ThreadPoolExecutor(max_workers=1)`. Batch requests with the same operation and kwargs until batch size or deadline. Complete futures on the event loop; reject `put_nowait` overflow; make `aclose()` idempotent and resolve or cancel every future.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_async_tokenizer.py`

Expected: all tests pass with no pending-task warnings.

- [ ] **Step 5: Commit**

```bash
git add auto_infer/serving/tokenizer.py tests/test_async_tokenizer.py
git commit -m "feat: add bounded async tokenizer"
```

### Task 3: Non-Blocking Engine-to-Async Output Bridge

**Files:**
- Modify: `auto_infer/serving/broker.py`
- Modify: `auto_infer/serving/service.py`
- Rewrite: `auto_infer/serving/async_engine.py`
- Modify: `tests/test_request_broker.py`
- Create: `tests/test_async_output_collector.py`

**Interfaces:**
- Produces: `AsyncOutputCollector(loop)`, `put_tokens(tuple[int, ...])`, `finish()`, `fail(error)`, `get()`, `AsyncRequestHandle`, `AsyncEngine.generate(ids, sampling)` async iterator, `abort(request_id)`, and `aclose()`.
- Preserves: the existing synchronous `EngineService.submit()` stream for IPC and current host tests.

- [ ] **Step 1: Write failing aggregation and cancellation tests**

```python
async def test_collector_merges_when_producer_gets_ahead():
    collector = AsyncOutputCollector(asyncio.get_running_loop())
    collector.put_tokens((1, 2))
    collector.put_tokens((3, 4))
    assert await collector.get() == (1, 2, 3, 4)
    assert collector.pending_slots == 0

async def test_async_generate_uses_no_to_thread(monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread",
                        lambda *a, **k: pytest.fail("blocking bridge used"))
    engine = AsyncEngine(_cfg(), MockExecutor(vocab_size=1000))
    assert [batch async for batch in engine.generate(
        [1, 2, 3], SamplingParams(max_tokens=3))] == [(4,), (5,), (6,)]
    await engine.aclose()
```

- [ ] **Step 2: Run tests and confirm current `to_thread` path fails**

Run: `pytest -q tests/test_async_output_collector.py tests/test_request_broker.py`

Expected: collector is missing and the current AsyncEngine calls `asyncio.to_thread`.

- [ ] **Step 3: Implement sink-based broker delivery**

Generalize RequestBroker entries to sinks with non-blocking `emit`, `finish`, and `fail`. Keep `ResponseQueue` as a compatibility sink. Add `AsyncOutputCollector` whose producer methods schedule one drain callback with `loop.call_soon_threadsafe`; merge token tuples into one pending slot and preserve terminal failure/completion.

- [ ] **Step 4: Rewrite AsyncEngine around async iteration**

Submit directly to an injected collector, yield token tuples from `collector.get()`, and issue exactly one abort in `finally`. `aclose()` rejects admission, aborts remaining handles, and closes EngineService without blocking the event loop indefinitely.

- [ ] **Step 5: Run serving-engine and IPC regressions**

Run: `pytest -q tests/test_async_output_collector.py tests/test_request_broker.py tests/test_serving_engine.py tests/test_serving_ipc.py`

Expected: all tests pass; legacy IPC streams remain operational.

- [ ] **Step 6: Commit**

```bash
git add auto_infer/serving/broker.py auto_infer/serving/service.py auto_infer/serving/async_engine.py tests/test_request_broker.py tests/test_async_output_collector.py
git commit -m "feat: bridge engine output to asyncio"
```

### Task 4: Incremental Text and Stop Processing

**Files:**
- Create: `auto_infer/serving/detokenizer.py`
- Create: `tests/test_incremental_detokenizer.py`

**Interfaces:**
- Produces: `IncrementalTextDecoder(tokenizer, stop, skip_special_tokens=True)`, `push(token_ids) -> TextDelta`, and `finish(reason) -> TextDelta`.
- `TextDelta` contains `text`, `token_count`, `finish_reason`, and `finished`.

- [ ] **Step 1: Write failing UTF-8, stop-boundary, and incremental-work tests**

```python
def test_stop_string_split_across_tokens_is_not_leaked():
    dec = IncrementalTextDecoder(PieceTokenizer({1: "hello<", 2: "STOP>tail"}),
                                 stop=["<STOP>"])
    assert dec.push((1,)).text == "hello"
    final = dec.push((2,))
    assert final.text == ""
    assert final.finish_reason == "stop"

def test_decode_work_is_suffix_bounded():
    tok = CountingTokenizer()
    dec = IncrementalTextDecoder(tok, stop=[])
    for token in range(100):
        dec.push((token,))
    assert tok.maximum_decode_input < 16
```

- [ ] **Step 2: Run tests and confirm module is missing**

Run: `pytest -q tests/test_incremental_detokenizer.py`

Expected: import failure.

- [ ] **Step 3: Implement bounded incremental decoding**

Maintain token pieces, prefix/read offsets, emitted character offset, maximum stop-string lookbehind, and pending UTF-8 replacement suffix. Never decode the full generated prefix after each token. Exclude EOS and caller stop-token IDs before calling the decoder.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_incremental_detokenizer.py`

Expected: all tests pass, including split stop strings and Unicode.

- [ ] **Step 5: Commit**

```bash
git add auto_infer/serving/detokenizer.py tests/test_incremental_detokenizer.py
git commit -m "feat: add incremental serving detokenizer"
```

### Task 5: Bounded Admission and Host-Side Metrics

**Files:**
- Create: `auto_infer/serving/admission.py`
- Rewrite: `auto_infer/serving/metrics.py`
- Create: `tests/test_admission.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Produces: `AdmissionController`, idempotent `AdmissionLease.release()`, `Overloaded`, `Unavailable`, and `ServingMetrics` with Prometheus rendering.
- Consumes: `ServingConfig`, prompt token counts, lifecycle health state.

- [ ] **Step 1: Write failing hard-limit and permit-reclamation tests**

```python
async def test_admission_has_hard_request_and_token_limits():
    gate = AdmissionController(max_http=2, max_waiting=1,
                               max_waiting_tokens=10)
    http = gate.acquire_http()
    engine = gate.acquire_engine(prompt_tokens=8)
    with pytest.raises(Overloaded):
        gate.acquire_engine(prompt_tokens=3)
    engine.release(); engine.release(); http.release()
    assert gate.snapshot().permits_in_use == 0

def test_prometheus_render_has_required_stages():
    text = ServingMetrics().render()
    for name in ("http_parse", "admission_wait", "tokenize", "engine_queue",
                 "prefill", "decode", "sse_send", "ttft", "itl", "e2e"):
        assert name in text
```

- [ ] **Step 2: Run tests and confirm missing classes fail**

Run: `pytest -q tests/test_admission.py tests/test_metrics.py`

Expected: import or attribute failures.

- [ ] **Step 3: Implement non-waiting admission leases and lifecycle state**

Use a lock-protected counter set so acquisition is atomic and immediately rejects overflow. Track HTTP inflight, engine waiting requests, and waiting prompt tokens. A lease owns the exact counters it decrements and ignores repeated release.

- [ ] **Step 4: Implement Prometheus metrics without device synchronization**

Use prometheus-client counters, gauges, and histograms. Expose a registry-local `render()` for tests and `/metrics`. Preserve the existing periodic `StatLogger` or adapt it to feed the same host counters so current engine tests do not lose observability.

- [ ] **Step 5: Run focused and existing metrics tests**

Run: `pytest -q tests/test_admission.py tests/test_metrics.py tests/test_spec_decode.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add auto_infer/serving/admission.py auto_infer/serving/metrics.py tests/test_admission.py tests/test_metrics.py
git commit -m "feat: bound serving admission and metrics"
```

### Task 6: FastAPI Text Serving Application

**Files:**
- Rewrite: `auto_infer/serving/api_server.py`
- Modify: `auto_infer/entrypoints/cli.py`
- Rewrite: `tests/test_api_server_architecture.py`
- Create: `tests/test_text_serving_api.py`

**Interfaces:**
- Produces: `ApiRuntime`, `create_app(runtime, serving_config) -> FastAPI`, async `serve(...)`, authenticated text endpoints, health, models, metrics, and SSE.
- Consumes: protocol models, AsyncTokenizer, AdmissionController, AsyncEngine, IncrementalTextDecoder, and ServingMetrics.

- [ ] **Step 1: Write failing app-contract and architecture tests**

```python
def test_architecture_is_native_async():
    source = inspect.getsource(api_server)
    assert "ThreadingHTTPServer" not in source
    assert "BaseHTTPRequestHandler" not in source
    assert "asyncio.to_thread" not in source

async def test_streaming_and_nonstreaming_match(app):
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as client:
        plain = await client.post("/v1/completions",
                                  json={"prompt": "abc", "max_tokens": 3})
        chunks = await collect_sse(client, "/v1/completions",
                                   {"prompt": "abc", "max_tokens": 3,
                                    "stream": True})
    assert join_text(chunks) == plain.json()["choices"][0]["text"]
    assert chunks[-1] == "[DONE]"
```

- [ ] **Step 2: Run tests and confirm the old HTTP server fails**

Run: `pytest -q tests/test_api_server_architecture.py tests/test_text_serving_api.py`

Expected: architecture assertion and missing app factory failures.

- [ ] **Step 3: Implement app factory and request pipeline**

Create FastAPI routes with dependency-based bearer authentication and shared runtime injected through `app.state`. Acquire/release both admission stages in `try/finally`. Use the async tokenizer, context validation, AsyncEngine iterator, and incremental decoder. Build non-streaming JSON and SSE from the same internal async generation function.

- [ ] **Step 4: Implement cancellation, errors, coalescing, and endpoints**

Abort on coroutine cancellation or disconnect. Return OpenAI-shaped `400`, `401`, `429`, `500`, and `503` responses. Emit the first SSE delta immediately, then coalesce up to the configured time/token threshold. Implement `/v1/models`, `/health`, and `/metrics` without touching device state.

- [ ] **Step 5: Wire uvicorn and CLI serving limits**

Add CLI flags for API key, HTTP inflight, waiting requests/tokens, tokenizer batch/capacity/wait, SSE coalescing, and shutdown grace. `serve()` constructs one runtime and invokes uvicorn; it never creates an engine per request.

- [ ] **Step 6: Run HTTP and serving regression tests**

Run: `pytest -q tests/test_api_server_architecture.py tests/test_text_serving_api.py tests/test_serving_engine.py tests/test_request_broker.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add auto_infer/serving/api_server.py auto_infer/entrypoints/cli.py tests/test_api_server_architecture.py tests/test_text_serving_api.py
git commit -m "feat: replace serving frontend with FastAPI"
```

### Task 7: Lifecycle, Fault, and Slow-Client Integration Gates

**Files:**
- Modify: `auto_infer/serving/api_server.py`
- Modify: `auto_infer/serving/async_engine.py`
- Modify: `tests/test_text_serving_api.py`
- Create: `tests/test_serving_lifecycle.py`
- Create: `tests/test_serving_load.py`

**Interfaces:**
- Verifies and completes startup readiness, graceful shutdown, recoverable/fatal engine state, disconnect abort, and slow-client isolation.

- [ ] **Step 1: Write failing lifecycle and overload tests**

```python
async def test_shutdown_stops_admission_then_aborts_after_grace(runtime):
    handle = await runtime.engine.start(long_request())
    await runtime.shutdown(grace_s=0)
    assert handle.aborted
    assert runtime.admission.snapshot().permits_in_use == 0
    assert not runtime.engine.service.thread.is_alive()

async def test_slow_client_does_not_grow_output_slots(runtime):
    slow = await runtime.engine.start(long_request())
    await asyncio.sleep(0.05)
    assert slow.collector.pending_slots <= 1
    fast = await collect(runtime.engine.generate(short_request()))
    assert fast
```

- [ ] **Step 2: Run tests and confirm missing lifecycle behavior**

Run: `pytest -q tests/test_serving_lifecycle.py tests/test_serving_load.py`

Expected: one or more lifecycle assertions fail before integration is completed.

- [ ] **Step 3: Complete readiness and ordered shutdown**

Make readiness true only after tokenizer and engine initialization. Shutdown closes admission, waits until the monotonic deadline, aborts remaining requests, closes collectors and tokenizer, then joins EngineService and closes the executor. Make repeated shutdown safe.

- [ ] **Step 4: Complete recoverable and fatal failure states**

Recoverable failures fail affected collectors, rebuild EngineCore, and restore health only after rebuild. Fatal failures atomically close admission, fail every collector, expose unhealthy status, and reject future submissions with `503` semantics.

- [ ] **Step 5: Run all host tests**

Run: `pytest -q`

Expected: full host suite passes with no leaked-task or thread warnings.

- [ ] **Step 6: Commit**

```bash
git add auto_infer/serving/api_server.py auto_infer/serving/async_engine.py tests/test_text_serving_api.py tests/test_serving_lifecycle.py tests/test_serving_load.py
git commit -m "test: gate serving lifecycle and overload"
```

### Task 8: Online Benchmark and NPU2 Production Validation

**Files:**
- Create: `benchmarks/run_serving_frontend.py`
- Create: `benchmarks/run_serving_online.py`
- Create: `scripts/verify_native_async_serving.py`
- Create: `scripts/soak_native_async_serving.py`
- Create: `docs/NATIVE-ASYNC-SERVING-VALIDATION-2026-07-20.md`

**Interfaces:**
- Produces reproducible JSON results for isolated frontend and Moonlight online tests plus a concise validation report.

- [ ] **Step 1: Add benchmark-manifest tests**

Add tests asserting benchmark outputs include workload identity, prompt/output lengths, arrival rate, concurrency, warmup, sample count, p50/p95/p99 TTFT/ITL/E2E, request/s, output tok/s, CPU, RSS, rejection rate, error rate, and git revision.

- [ ] **Step 2: Implement isolated frontend benchmark**

Use an injectable deterministic dummy async engine and localhost HTTP clients to compare auto-infer and vLLM frontend overhead under identical B1/B4/B16 workloads. Record raw samples and aggregate percentiles; do not claim superiority without raw artifacts.

- [ ] **Step 3: Implement Moonlight online benchmark and stability runner**

Drive identical prompt corpora, lengths, sampling, arrival rates, and warmup against auto-infer, vllm-ascend, and omni-npu. Include slow-client and overload phases. The soak runner continuously verifies response correctness, queue bounds, process liveness, RSS plateau, and KV reclamation.

- [ ] **Step 4: Run host benchmark smoke tests**

Run: `pytest -q tests/test_benchmark_manifest.py && python benchmarks/run_serving_frontend.py --smoke`

Expected: tests pass and a schema-complete smoke JSON file is produced.

- [ ] **Step 5: Deploy to NPU2 `/data2` and run correctness gates**

Synchronize the committed worktree without deleting unrelated remote data. Run the full host suite in the target environment, start Moonlight Serving, verify greedy online/offline token identity, stream/non-stream identity, B1/B4/B16 continuous batching, cancellation, overload, and fault recovery.

- [ ] **Step 6: Run performance and soak gates**

Run isolated frontend comparisons, then real Moonlight online comparisons. Run the 24-hour soak only after correctness passes. Require throughput coefficient of variation at most 3%, no sustained memory growth, and no request/KV/thread leakage. Under the slow-client phase, unaffected-client throughput regression must be at most 5%.

- [ ] **Step 7: Write evidence-backed validation report**

Record exact commands, revisions, environment, raw artifact paths, pass/fail status, and limitations. State frontend superiority only if the isolated benchmark supports it; state full-system superiority only if the aligned online benchmark supports it.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/run_serving_frontend.py benchmarks/run_serving_online.py scripts/verify_native_async_serving.py scripts/soak_native_async_serving.py docs/NATIVE-ASYNC-SERVING-VALIDATION-2026-07-20.md tests/test_benchmark_manifest.py
git commit -m "bench: validate native async serving"
```

## Final Verification

- [ ] Run `git diff --check` and `pytest -q`.
- [ ] Confirm `rg -n "ThreadingHTTPServer|BaseHTTPRequestHandler|asyncio.to_thread" auto_infer/serving` returns no Serving request-path matches.
- [ ] Confirm every in-scope endpoint, error status, cancellation path, queue bound, and lifecycle state has a host test.
- [ ] Confirm NPU2 correctness gates pass before accepting performance or soak results.
- [ ] Confirm the worktree contains no unrelated changes.
