"""Hard, non-waiting Serving admission limits with idempotent leases."""

import threading
from dataclasses import dataclass
from typing import Callable


class Overloaded(RuntimeError):
    pass


class Unavailable(RuntimeError):
    pass


class AdmissionLease:
    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


@dataclass(frozen=True)
class AdmissionSnapshot:
    open: bool
    http_inflight: int
    engine_requests: int
    engine_tokens: int

    @property
    def permits_in_use(self) -> int:
        return self.http_inflight + self.engine_requests


class AdmissionController:
    def __init__(self, *, max_http: int, max_engine_requests: int,
                 max_engine_tokens: int):
        limits = {
            "max_http": max_http,
            "max_engine_requests": max_engine_requests,
            "max_engine_tokens": max_engine_tokens,
        }
        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        self.max_http = max_http
        self.max_engine_requests = max_engine_requests
        self.max_engine_tokens = max_engine_tokens
        self._lock = threading.Lock()
        self._open = True
        self._http_inflight = 0
        self._engine_requests = 0
        self._engine_tokens = 0

    def acquire_http(self) -> AdmissionLease:
        with self._lock:
            self._require_open()
            if self._http_inflight >= self.max_http:
                raise Overloaded("HTTP request capacity is full")
            self._http_inflight += 1
        return AdmissionLease(self._release_http)

    def acquire_engine(self, *, prompt_tokens: int) -> AdmissionLease:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be > 0")
        with self._lock:
            self._require_open()
            if self._engine_requests >= self.max_engine_requests:
                raise Overloaded("engine request capacity is full")
            if self._engine_tokens + prompt_tokens > self.max_engine_tokens:
                raise Overloaded("engine token capacity is full")
            self._engine_requests += 1
            self._engine_tokens += prompt_tokens
        return AdmissionLease(
            lambda: self._release_engine(prompt_tokens)
        )

    def close(self) -> None:
        with self._lock:
            self._open = False

    def open(self) -> None:
        with self._lock:
            self._open = True

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            return AdmissionSnapshot(
                open=self._open,
                http_inflight=self._http_inflight,
                engine_requests=self._engine_requests,
                engine_tokens=self._engine_tokens,
            )

    def _require_open(self) -> None:
        if not self._open:
            raise Unavailable("admission is closed")

    def _release_http(self) -> None:
        with self._lock:
            self._http_inflight -= 1

    def _release_engine(self, prompt_tokens: int) -> None:
        with self._lock:
            self._engine_requests -= 1
            self._engine_tokens -= prompt_tokens
