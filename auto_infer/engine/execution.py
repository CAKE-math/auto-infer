from dataclasses import dataclass, field
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Mapping

import torch

from auto_infer.engine.scheduler import Scheduler, SchedulerOutput, ScheduledRequest


class TokenSpan(Sequence):
    """Fixed-length read-only window over request-owned token storage."""

    def __init__(self, source, length=None):
        self.source = source
        self.length = len(source) if length is None else length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return tuple(self.source[start:stop:step])
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source[index]

    def __eq__(self, other):
        return tuple(self) == tuple(other)


class TokenView(Sequence):
    """Allocation-free prompt/output concatenation with snapshotted lengths."""

    def __init__(self, prompt_source, output_source):
        self.prompt_source = prompt_source
        self.output_source = output_source
        self._prompt_length = len(prompt_source)
        self._output_length = len(output_source)

    def __len__(self):
        return self._prompt_length + self._output_length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position]
                    for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self._prompt_length:
            return self.prompt_source[index]
        return self.output_source[index - self._prompt_length]


@dataclass(frozen=True)
class DeviceTokenRef:
    """One row in a retained batch; the owner keeps tensor storage alive."""
    owner: "DeviceTokenBatch"
    row: int


@dataclass(frozen=True)
class DeviceTokenBatch:
    """Immutable ownership metadata around one complete sampled-token tensor."""
    tokens: torch.Tensor
    order: tuple[str, ...]
    row_by_request: Mapping[str, int]

    @classmethod
    def from_output(cls, tokens: torch.Tensor, order) -> "DeviceTokenBatch":
        order = tuple(order)
        if tokens.ndim != 1 or tokens.shape[0] != len(order):
            raise ValueError("sampled tokens and request order must have equal length")
        rows = {rid: row for row, rid in enumerate(order)}
        if len(rows) != len(order):
            raise ValueError("sampled request order contains duplicate IDs")
        return cls(tokens=tokens, order=order,
                   row_by_request=MappingProxyType(rows))

    def refs(self) -> dict[str, DeviceTokenRef]:
        return {rid: DeviceTokenRef(self, row)
                for rid, row in self.row_by_request.items()}


@dataclass(frozen=True)
class SamplingView:
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    frequency_penalty: float
    repetition_penalty: float
    logit_bias: Mapping[int, float] | None
    bad_words_token_ids: tuple[tuple[int, ...], ...] | None
    allowed_token_ids: tuple[int, ...] | None
    min_tokens: int
    ignore_eos: bool
    eos_token_id: int | None
    stop_token_ids: tuple[int, ...]
    seed: int | None

    @classmethod
    def from_params(cls, params):
        bias = None if params.logit_bias is None else MappingProxyType(dict(params.logit_bias))
        bad = (None if params.bad_words_token_ids is None else
               tuple(tuple(row) for row in params.bad_words_token_ids))
        allowed = None if params.allowed_token_ids is None else tuple(params.allowed_token_ids)
        return cls(
            max_tokens=params.max_tokens, temperature=params.temperature,
            top_k=params.top_k, top_p=params.top_p, min_p=params.min_p,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty, logit_bias=bias,
            bad_words_token_ids=bad, allowed_token_ids=allowed,
            min_tokens=params.min_tokens, ignore_eos=params.ignore_eos,
            eos_token_id=params.eos_token_id,
            stop_token_ids=tuple(params.stop_token_ids), seed=params.seed)


@dataclass(frozen=True)
class RequestView:
    request_id: str
    prompt_token_ids: TokenSpan
    output_token_ids: TokenSpan
    token_ids: TokenView
    sampling: SamplingView
    num_computed_tokens: int
    num_prefill_tokens: int
    spec_draft: tuple[int, ...]

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> TokenView:
        return self.token_ids


@dataclass(frozen=True)
class BatchPlan:
    scheduled: tuple[ScheduledRequest, ...]
    num_batched_tokens: int
    requests: Mapping[str, RequestView]
    block_tables: Mapping[str, tuple[int, ...]]

    @classmethod
    def from_scheduler(cls, output: SchedulerOutput, scheduler: Scheduler):
        requests = {}
        tables = {}
        for scheduled in output.scheduled:
            req = scheduler.get_request(scheduled.request_id)
            token_ids = TokenView(req.prompt_token_ids, req.output_token_ids)
            requests[req.request_id] = RequestView(
                request_id=req.request_id,
                prompt_token_ids=TokenSpan(
                    req.prompt_token_ids, len(req.prompt_token_ids)),
                output_token_ids=TokenSpan(
                    req.output_token_ids, len(req.output_token_ids)),
                token_ids=token_ids,
                sampling=SamplingView.from_params(req.sampling),
                num_computed_tokens=req.num_computed_tokens,
                num_prefill_tokens=req.num_prefill_tokens,
                spec_draft=tuple(req.spec_draft))
            tables[req.request_id] = tuple(scheduler.block_tables.get(req.request_id, ()))
        return cls(tuple(output.scheduled), output.num_batched_tokens,
                   MappingProxyType(requests), MappingProxyType(tables))

    def get_request(self, request_id: str) -> RequestView:
        return self.requests[request_id]


@dataclass(frozen=True)
class ExecutionStats:
    accepted: int = 0
    steps: int = 0
    accepted_per_position: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    tokens: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    next_drafts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    errors: Mapping[str, str] = field(default_factory=dict)
    stats: ExecutionStats = field(default_factory=ExecutionStats)

    @classmethod
    def from_single_tokens(cls, tokens: Mapping[str, Any]):
        return cls(tokens={rid: (int(token),) for rid, token in tokens.items()})

    def single_tokens(self) -> dict[str, int]:
        return {rid: values[0] for rid, values in self.tokens.items() if values}
