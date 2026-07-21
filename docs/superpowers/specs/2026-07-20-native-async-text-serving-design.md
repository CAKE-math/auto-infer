# Native Async Text Serving Design

Date: 2026-07-20
Status: Approved design

## Objective

Build a production-grade text-generation Serving layer whose functional coverage
matches the useful core of vLLM Serving while remaining independent of vLLM's
implementation. The Serving path should be simpler and no slower than vLLM's;
auto-infer's competitive advantage remains in EngineCore, continuous batching,
KV-cache management, graph execution, and MTP.

This is functional parity, not source compatibility or a promise to track every
vLLM release.

## Scope

The supported HTTP surface is deliberately small:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

Both generation endpoints support streaming and non-streaming responses,
sampling parameters, stop strings, stop-token IDs, usage accounting, structured
errors, authentication, request cancellation, and continuous batching.

The following are explicitly out of scope: embeddings, pooling, reranking,
Responses API, dynamic LoRA loading, audio, transcription, Realtime, and other
multimodal or administrative APIs.

## Chosen Approach

Use a native asynchronous HTTP frontend connected to a dedicated EngineService
thread through bounded, non-blocking control and output bridges.

Two alternatives were rejected:

1. Wrapping the existing blocking queues with `asyncio.to_thread()` is smaller,
   but consumes a worker thread per blocked stream and leaves the slow-client
   path structurally weak.
2. Requiring a separate Engine process provides stronger fault isolation but
   adds IPC latency and operational complexity to every deployment. Process
   transport remains an optional production-topology adapter, not the base
   single-instance architecture.

## Architecture

```text
FastAPI / uvicorn
        |
        v
Protocol validation, authentication, admission control
        |
        v
Asynchronous micro-batched tokenizer
        |
        v
AsyncEngineClient
        | bounded control queue
        v
EngineService thread
        |
        v
EngineCore / continuous batching / NPU executor
        |
        v
Single-slot aggregating OutputCollector
        |
        v
Incremental detokenizer / SSE encoder
```

EngineCore remains single-owner. HTTP handlers, tokenizer work, metrics
collection, and network writes never enter the scheduler or device hot path.

## Components and Ownership

### Protocol layer

Typed request models validate supported parameters before engine admission.
Response builders own OpenAI-shaped JSON and SSE payloads. Protocol types do
not import EngineCore or NPU code.

### Admission controller

Admission has two bounded stages:

- an HTTP/tokenizer concurrency gate protects CPU and event-loop capacity;
- an engine-waiting gate limits queued requests and queued prompt tokens.

Permits are released exactly once on completion, rejection, cancellation, or
failure. Queue saturation is rejected rather than allowed to grow memory.

### Async tokenizer

Tokenization and chat-template rendering run in a dedicated executor. Compatible
requests are micro-batched. The default batch size is 32 and the default batch
wait is 2 ms. Its input queue is bounded, and shutdown drains or cancels every
pending future.

### AsyncEngineClient

The client submits immutable request commands through a bounded multi-producer,
single-consumer queue. EngineService owns EngineCore and is the only component
allowed to call `add_request`, `step`, `abort`, recovery, or executor shutdown.

The engine thread publishes results with `loop.call_soon_threadsafe()`. It never
blocks on an asyncio consumer or a socket.

### OutputCollector

Each request has one pending aggregate rather than a per-token queue. If the
producer outruns the consumer, new deltas merge into the pending aggregate.
Completion and failure signals cannot be overwritten. This makes slow clients
independent of engine progress without an unbounded message backlog.

### Incremental detokenizer and SSE encoder

Detokenization maintains token, prefix, and read offsets and decodes only the
new suffix. UTF-8 fragments and stop strings are held until their status is
unambiguous.

The first visible token is sent immediately. Subsequent deltas may be coalesced
by token count or a configurable time window, defaulting to 5 ms. Coalescing
does not delay the terminal event.

### Metrics

Metrics use host-side timestamps and counters only. Collecting metrics must not
synchronize the NPU. Required observations are HTTP parsing, admission wait,
tokenization, engine queue, prefill, decode, SSE serialization/send, TTFT, ITL,
E2E latency, throughput, running/waiting counts, admission rejection, aborts,
errors, KV utilization, and process CPU/RSS.

## Request Data Flow

1. FastAPI parses and validates the request and authenticates the caller.
2. The HTTP admission gate is acquired or the request is rejected.
3. The async tokenizer renders chat input and produces token IDs.
4. Context length and token-budget constraints are validated.
5. Engine admission is acquired and AsyncEngineClient submits the request.
6. EngineService admits requests into the existing continuous-batching
   scheduler and steps EngineCore.
7. Generated token deltas are committed to the request's OutputCollector.
8. The HTTP coroutine incrementally detokenizes and emits JSON or SSE.
9. Every exit path releases the engine request, collector, and admission permits.

Streaming and non-streaming requests share all steps through output collection;
only the final response consumer differs.

## Errors and Cancellation

- Invalid parameters, context length, or sampling combinations return a
  structured `400` response.
- Missing or invalid credentials return `401`.
- A full admission queue returns `429`.
- An unhealthy or shutting-down engine returns `503`.
- Unexpected pre-header failures return structured `500` responses.
- A failure after SSE headers emits a structured error event and closes the
  stream.
- Client disconnect immediately issues an idempotent engine abort and releases
  all owned resources.
- Recoverable executor failures fail affected requests, rebuild EngineCore, and
  reopen admission only after health is restored.
- Non-recoverable failures close admission, fail all active requests, mark
  health unavailable, and do not enter an infinite restart loop.

## Lifecycle

Startup completes only after the tokenizer, EngineService, executor, model, and
health state are ready. Shutdown is idempotent and ordered:

1. stop new admission;
2. allow active requests to finish within a configurable grace period, default
   30 seconds;
3. abort remaining requests;
4. close collectors and pending tokenizer work;
5. stop EngineService and close the executor;
6. stop HTTP serving.

Every queue, task, executor, thread, and engine object has one documented owner.

## Default Limits

- tokenizer micro-batch size: 32
- tokenizer micro-batch wait: 2 ms
- SSE coalescing window: 5 ms, excluding the first token
- maximum waiting requests: `2 * max_num_seqs`
- graceful shutdown: 30 seconds

All limits are configurable. Invalid or non-positive limits fail during startup.

## Correctness Tests

- Contract tests cover every in-scope endpoint and structured error.
- Greedy online token IDs exactly match direct offline Engine output.
- Concatenated SSE content exactly matches the non-streaming response.
- Tests cover EOS, stop tokens, stop strings crossing token boundaries,
  `max_tokens`, UTF-8 boundaries, usage, and finish reason.
- Invalid parameters, oversized contexts, authentication, duplicate IDs, and
  engine failures have deterministic outcomes.
- Architecture tests prohibit `ThreadingHTTPServer`, request-local engine
  construction, per-request blocking worker threads, and full-prefix decode per
  emitted token.

## Concurrency and Stability Tests

- Mixed prompt/output lengths at B1, B4, B16, burst traffic, and sustained load.
- Explicit proof that concurrent requests share continuous-batching steps.
- Slow-client, disconnect, cancellation, overload, tokenizer-failure, and
  engine-failure injection.
- Queue depths never exceed configured bounds, and overload memory reaches a
  plateau.
- Completion, cancellation, and failure reclaim KV blocks, collectors, futures,
  and admission permits.
- Moonlight runs for 24 hours on NPU2 without request loss, deadlock, thread
  leakage, KV leakage, or sustained memory growth.
- Steady-state throughput coefficient of variation is at most 3%.

## Performance Validation

Use two complementary comparisons:

1. A dummy-engine benchmark isolates HTTP, tokenizer, collector, detokenizer,
   and SSE overhead and compares the frontend directly with vLLM on the same
   host.
2. A real Moonlight NPU benchmark compares full online systems using identical
   prompts, output lengths, sampling, arrival rates, warmup, and concurrency.

Report p50/p95/p99 TTFT, ITL, and E2E latency, request throughput, output tok/s,
CPU, RSS, rejection rate, and error rate. The first token must never wait for
the SSE coalescing window. Introducing slow clients must reduce unaffected-client
throughput by no more than 5%. Offline Engine performance must not regress.

Acceptance requires complete in-scope functional coverage, frontend overhead no
higher than vLLM under the isolated benchmark, and full-system online performance
that preserves auto-infer's Engine advantage.
