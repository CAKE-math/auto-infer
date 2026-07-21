"""Configuration for the native asynchronous text Serving boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingConfig:
    max_num_seqs: int = 256
    max_http_inflight: int = 512
    max_waiting_requests: int | None = None
    max_waiting_tokens: int = 1_048_576
    tokenizer_batch_size: int = 32
    tokenizer_queue_capacity: int = 1024
    tokenizer_wait_ms: float = 2.0
    sse_coalesce_ms: float = 5.0
    sse_coalesce_tokens: int = 8
    shutdown_grace_s: float = 30.0
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.max_waiting_requests is None:
            object.__setattr__(
                self, "max_waiting_requests", 2 * self.max_num_seqs
            )
        positive = (
            "max_num_seqs",
            "max_http_inflight",
            "max_waiting_requests",
            "max_waiting_tokens",
            "tokenizer_batch_size",
            "tokenizer_queue_capacity",
            "sse_coalesce_tokens",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        non_negative = (
            "tokenizer_wait_ms",
            "sse_coalesce_ms",
            "shutdown_grace_s",
        )
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
