"""Host-known selection between captured greedy and general decode sampling."""

import torch


def stable_greedy_argmax(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    candidate_count: int = 4,
) -> torch.Tensor:
    """Re-score the BF16 top candidates with a transient FP32 dot product.

    This avoids materializing a full FP32 vocabulary head while preventing a
    BF16 logit-bin tie from choosing different tokens across equivalent eager
    and ACL Graph kernels.
    """
    if candidate_count <= 0:
        raise ValueError("candidate_count must be > 0")
    count = min(candidate_count, logits.shape[-1])
    candidates = torch.topk(logits, count, dim=-1).indices
    rows = weight.index_select(0, candidates.flatten()).view(
        *candidates.shape, hidden.shape[-1])
    scores = torch.sum(
        rows.float() * hidden.float().unsqueeze(-2), dim=-1)
    best = scores.max(dim=-1, keepdim=True).values
    sampled = torch.where(
        scores == best, candidates, logits.shape[-1]).min(dim=-1).values
    if out is not None:
        out.copy_(sampled)
        return out
    return sampled


def is_capturable_greedy(requests) -> bool:
    """Whether requests may use the captured stable-greedy epilogue."""
    if not requests:
        return False
    for request in requests:
        params = request.sampling
        if params.temperature > 0.0:
            return False
        if (params.presence_penalty or params.frequency_penalty
                or params.repetition_penalty != 1.0):
            return False
        if params.logit_bias or params.allowed_token_ids is not None:
            return False
        if any(len(group) == 1 for group in (params.bad_words_token_ids or ())):
            return False
        stop_is_blocked = (
            len(request.output_token_ids) < params.min_tokens
            or params.ignore_eos)
        if stop_is_blocked and params.stop_token_ids:
            return False
    return True
