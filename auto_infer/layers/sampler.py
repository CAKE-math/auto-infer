"""Sampling (spec sec 8). Graph-capturable ops: greedy / temperature / top-k /
top-p / min-p, plus a fully vectorized batched path (`sample_batched` +
`SamplingTensors`) covering per-row penalties (repetition / frequency /
presence), logit bias, and disallowed-token masks. No host control flow inside
the sampling math (mask-based), so it can be captured into the decode graph.
Stop-strings / n>1 are follow-ups.
"""
from dataclasses import dataclass

import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """logits: (..., vocab) -> token ids (...)."""
    return logits.argmax(dim=-1)


def sample(logits: torch.Tensor, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0,
           generator: "torch.Generator | None" = None) -> torch.Tensor:
    """Graph-capturable sampling. temperature<=0 -> greedy.
    logits: (N, vocab) -> token ids (N,)."""
    if temperature <= 0.0:
        return greedy(logits)
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # keep tokens up to and including the one crossing top_p
        remove = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@dataclass
class SamplingTensors:
    temperature: torch.Tensor              # (B,)
    top_k: torch.Tensor                    # (B,) long, 0 = disabled
    top_p: torch.Tensor                    # (B,)
    min_p: torch.Tensor                    # (B,)
    presence: torch.Tensor                 # (B,)
    frequency: torch.Tensor                # (B,)
    repetition: torch.Tensor               # (B,)
    occurrence_counts: "torch.Tensor | None"   # (B, vocab) float counts of output tokens
    prompt_presence: "torch.Tensor | None"     # (B, vocab) 1.0 where token in prompt/output
    bias: "torch.Tensor | None"                # (B, vocab) additive
    disallowed_mask: "torch.Tensor | None"     # (B, vocab) bool, True = forbid
    # Host-known guard flags (no device access needed to compute these — they
    # come straight from the Python SamplingParams). When None, sample_batched
    # falls back to the device-sync computation (used by tests that construct
    # SamplingTensors directly). Set them from build_sampling_tensors to avoid
    # device->host syncs in sample_batched, which would break async submit()
    # on the NPU path (sync must happen only in collect()).
    has_top_k: "bool | None" = None
    has_top_p: "bool | None" = None
    has_min_p: "bool | None" = None
    all_greedy_unprocessed: bool = False


def _apply_penalties(logits, t):
    if t.occurrence_counts is not None:
        # repetition: positive logits / rep, negative * rep, only for seen tokens
        seen = (t.occurrence_counts > 0)
        if t.prompt_presence is not None:
            seen = seen | (t.prompt_presence > 0)
        rep = t.repetition.unsqueeze(-1)
        repd = torch.where(logits > 0, logits / rep, logits * rep)
        logits = torch.where(seen, repd, logits)
        logits = logits - t.frequency.unsqueeze(-1) * t.occurrence_counts
        presence_hit = seen.to(logits.dtype)
        logits = logits - t.presence.unsqueeze(-1) * presence_hit
    return logits


def sample_batched(logits: torch.Tensor, t: SamplingTensors,
                   generator: "torch.Generator | None" = None) -> torch.Tensor:
    """logits (B, vocab) -> token ids (B,). Fully vectorized, mask/arith only."""
    if t.all_greedy_unprocessed:
        return logits.argmax(dim=-1)
    logits = logits.float()
    if t.bias is not None:
        logits = logits + t.bias
    logits = _apply_penalties(logits, t)
    if t.disallowed_mask is not None:
        logits = logits.masked_fill(t.disallowed_mask, float("-inf"))

    greedy_rows = t.temperature <= 0.0
    temp = torch.where(greedy_rows, torch.ones_like(t.temperature), t.temperature)
    scaled = logits / temp.unsqueeze(-1)

    # Guard conditions are known host-side from the Python SamplingParams
    # (see build_sampling_tensors). Checking t.has_* avoids a device->host
    # sync (int()/float() on a device tensor) inside sample_batched, which
    # would otherwise force a wait on the async NPU submit() path. Fall back
    # to the device computation only when the flag wasn't provided (e.g.
    # tests that construct SamplingTensors directly).
    use_top_k = t.has_top_k if t.has_top_k is not None else (int(t.top_k.max()) > 0)
    use_top_p = t.has_top_p if t.has_top_p is not None else (float(t.top_p.min()) < 1.0)
    use_min_p = t.has_min_p if t.has_min_p is not None else (float(t.min_p.max()) > 0.0)

    # top_k per row (0 = disabled): mask everything below the row's k-th value
    if use_top_k:
        vocab = scaled.shape[-1]
        kcap = torch.where(t.top_k > 0, t.top_k, torch.full_like(t.top_k, vocab))
        kcap = kcap.clamp(max=vocab)
        sorted_vals, _ = torch.sort(scaled, descending=True, dim=-1)
        kth = sorted_vals.gather(-1, (kcap - 1).clamp(min=0).unsqueeze(-1))
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

    if use_top_p:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) > t.top_p.unsqueeze(-1)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, sorted_idx, sorted_logits)

    if use_min_p:
        probs = torch.softmax(scaled, dim=-1)
        top = probs.max(dim=-1, keepdim=True).values
        scaled = torch.where(probs < t.min_p.unsqueeze(-1) * top,
                             torch.full_like(scaled, float("-inf")), scaled)

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    greedy_tok = logits.argmax(dim=-1)
    return torch.where(greedy_rows, greedy_tok, sampled)
