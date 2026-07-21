from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: dict[int, float] | None = None
    bad_words_token_ids: list[list[int]] | None = None   # pre-tokenized (no str->tok here)
    allowed_token_ids: list[int] | None = None
    min_tokens: int = 0
    ignore_eos: bool = False
    eos_token_id: int | None = None                      # model EOS; stops decode unless ignore_eos
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must satisfy 0 <= min_tokens <= max_tokens")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must satisfy 0 <= min_p <= 1")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    priority: int = 0
    num_prefill_tokens: int = -1     # tokens to prefill before decode (recompute sets this)
    arrival_time: float | None = None    # set by EngineCore.add_request (for TTFT)
    first_token_time: float | None = None  # set when the first output token is produced
    spec_draft: list[int] = field(default_factory=list)  # spec-decode: this step's k drafts

    def __post_init__(self):
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.num_prefill_tokens < 0:
            self.num_prefill_tokens = len(self.prompt_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        sp = self.sampling
        n = len(self.output_token_ids)
        if n >= sp.max_tokens:                           # length cap always wins
            return True
        if n == 0 or n < sp.min_tokens:                  # honor min_tokens before any stop token
            return False
        last = self.output_token_ids[-1]
        if not sp.ignore_eos and sp.eos_token_id is not None and last == sp.eos_token_id:
            return True                                  # model EOS
        return last in sp.stop_token_ids                 # caller-supplied stop tokens
