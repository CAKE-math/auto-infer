"""Typed HTTP protocol objects kept independent from EngineCore internals."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_infer.engine.request import SamplingParams


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(_StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class _GenerationRequest(_StrictModel):
    model: str | None = None
    max_tokens: int = Field(default=16, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    logit_bias: dict[int, float] | None = None
    bad_words_token_ids: list[list[int]] | None = None
    allowed_token_ids: list[int] | None = None
    min_tokens: int = Field(default=0, ge=0)
    ignore_eos: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False

    @model_validator(mode="after")
    def validate_token_limits(self):
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
        return self


class CompletionRequest(_GenerationRequest):
    prompt: str = Field(min_length=1)


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)


def sampling_params(
    request: _GenerationRequest, eos_token_id: int | None
) -> SamplingParams:
    """Translate the public protocol once at the Engine boundary."""
    return SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty,
        logit_bias=request.logit_bias,
        bad_words_token_ids=request.bad_words_token_ids,
        allowed_token_ids=request.allowed_token_ids,
        min_tokens=request.min_tokens,
        ignore_eos=request.ignore_eos,
        eos_token_id=eos_token_id,
        stop_token_ids=list(request.stop_token_ids),
        seed=request.seed,
    )


def stop_strings(request: _GenerationRequest) -> tuple[str, ...]:
    if request.stop is None:
        return ()
    if isinstance(request.stop, str):
        return (request.stop,)
    return tuple(request.stop)


def error_payload(
    message: str,
    *,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
