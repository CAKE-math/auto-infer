import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class _Failure:
    error: BaseException


class ResponseQueue(queue.Queue):
    def get(self, *args, **kwargs):
        item = super().get(*args, **kwargs)
        if isinstance(item, _Failure):
            raise RuntimeError(f"engine request failed: {item.error}") from item.error
        return item

    def emit(self, tokens: tuple[int, ...]) -> None:
        for token in tokens:
            self.put(token)

    def finish(self) -> None:
        self.put(None)

    def fail(self, error: BaseException) -> None:
        self.put(_Failure(error))


class ResponseSink(Protocol):
    def emit(self, tokens: tuple[int, ...]) -> None: ...

    def finish(self) -> None: ...

    def fail(self, error: BaseException) -> None: ...


class AsyncOutputCollector:
    """One aggregating output slot owned by an asyncio event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._tokens: list[int] = []
        self._finished = False
        self._error: BaseException | None = None
        self._terminal_consumed = False
        self._ready = asyncio.Event()
        self._producer_lock = threading.Lock()
        self._producer_tokens: list[int] = []
        self._producer_finished = False
        self._producer_error: BaseException | None = None
        self._notification_scheduled = False

    @property
    def pending_slots(self) -> int:
        with self._producer_lock:
            producer_pending = bool(
                self._producer_tokens or self._producer_finished
                or self._producer_error is not None
            )
        return int(
            producer_pending or bool(self._tokens) or self._finished
            or self._error is not None
        )

    def emit(self, tokens: tuple[int, ...]) -> None:
        self.put_tokens(tokens)

    def put_tokens(self, tokens: tuple[int, ...]) -> None:
        if not tokens:
            return
        with self._producer_lock:
            if self._producer_finished or self._producer_error is not None:
                return
            self._producer_tokens.extend(tokens)
            self._schedule_notification_locked()

    def finish(self) -> None:
        with self._producer_lock:
            if self._producer_finished or self._producer_error is not None:
                return
            self._producer_finished = True
            self._schedule_notification_locked()

    def fail(self, error: BaseException) -> None:
        with self._producer_lock:
            if self._producer_finished or self._producer_error is not None:
                return
            self._producer_error = error
            self._schedule_notification_locked()

    def _schedule_notification_locked(self) -> None:
        if self._notification_scheduled:
            return
        self._notification_scheduled = True
        self._loop.call_soon_threadsafe(self._drain_producer)

    def _drain_producer(self) -> None:
        with self._producer_lock:
            tokens = tuple(self._producer_tokens)
            self._producer_tokens.clear()
            finished = self._producer_finished
            error = self._producer_error
            self._notification_scheduled = False
        if error is not None:
            self._commit_failure(error)
            return
        if tokens:
            self._commit_tokens(tokens)
        if finished:
            self._commit_finish()

    def _commit_tokens(self, tokens: tuple[int, ...]) -> None:
        if not self._finished and self._error is None:
            self._tokens.extend(tokens)
            self._ready.set()

    def _commit_finish(self) -> None:
        if self._error is None:
            self._finished = True
            self._ready.set()

    def _commit_failure(self, error: BaseException) -> None:
        self._tokens.clear()
        self._error = error
        self._ready.set()

    async def get(self) -> tuple[int, ...] | None:
        while not self._tokens and not self._finished and self._error is None:
            self._ready.clear()
            await self._ready.wait()
        if self._error is not None:
            error = self._error
            self._error = None
            self._terminal_consumed = True
            self._ready.clear()
            raise RuntimeError(f"engine request failed: {error}") from error
        if self._tokens:
            tokens = tuple(self._tokens)
            self._tokens.clear()
            if not self._finished:
                self._ready.clear()
            return tokens
        self._terminal_consumed = True
        self._ready.clear()
        return None


class RequestBroker:
    """Thread-safe owner of per-request response streams."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sinks: dict[str, ResponseSink] = {}
        self._closed = False

    def create(self, request_id: str) -> ResponseQueue:
        with self._lock:
            if self._closed:
                raise RuntimeError("engine service is closed")
            if request_id in self._sinks:
                raise ValueError(f"duplicate request id: {request_id}")
            stream = ResponseQueue()
            self._sinks[request_id] = stream
            return stream

    def register(self, request_id: str, sink: ResponseSink) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("engine service is closed")
            if request_id in self._sinks:
                raise ValueError(f"duplicate request id: {request_id}")
            self._sinks[request_id] = sink

    def emit(self, request_id: str, tokens: tuple[int, ...]) -> None:
        with self._lock:
            sink = self._sinks.get(request_id)
        if sink is not None:
            sink.emit(tokens)

    def finish(self, request_id: str) -> None:
        with self._lock:
            sink = self._sinks.pop(request_id, None)
        if sink is not None:
            sink.finish()

    def fail(self, request_id: str, error: BaseException) -> None:
        with self._lock:
            sink = self._sinks.pop(request_id, None)
        if sink is not None:
            sink.fail(error)

    def fail_all(self, error: BaseException) -> None:
        with self._lock:
            items = list(self._sinks.values())
            self._sinks.clear()
        for sink in items:
            sink.fail(error)

    def cancel(self, request_id: str) -> None:
        # Wake either compatibility queue consumers or native async collectors.
        self.finish(request_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            items = list(self._sinks.values())
            self._sinks.clear()
        for sink in items:
            sink.finish()
