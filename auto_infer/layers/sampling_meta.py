"""Host-side construction of batched SamplingTensors from scheduled requests."""
import torch

from auto_infer.layers.sampler import SamplingTensors


def build_sampling_tensors(reqs, vocab: int, device):
    """reqs: list of Request (one per sampled row, in output order).
    Returns (SamplingTensors on `device`, order: list[request_id])."""
    B = len(reqs)
    order = [r.request_id for r in reqs]
    sp = [r.sampling for r in reqs]

    def col(attr, default, dtype=torch.float32):
        return torch.tensor([getattr(p, attr, default) for p in sp], dtype=dtype, device=device)

    temperature = col("temperature", 0.0)
    top_k = col("top_k", 0, dtype=torch.long)
    top_p = col("top_p", 1.0)
    min_p = col("min_p", 0.0)
    presence = col("presence_penalty", 0.0)
    frequency = col("frequency_penalty", 0.0)
    repetition = col("repetition_penalty", 1.0)

    # occurrence_counts / prompt_presence are expensive (B, vocab) tensors
    # built with per-token Python loops, so only build them when some request
    # in the batch actually has an active penalty. Always build BOTH together
    # or NEITHER — sample_batched's `_apply_penalties` branches on
    # `occurrence_counts is not None` alone, so a lone `prompt_presence`
    # would be silently ignored while a lone `occurrence_counts` would crash
    # (or vice versa) the moment the other is expected.
    need_pen = any(p.presence_penalty or p.frequency_penalty or (p.repetition_penalty != 1.0)
                   for p in sp)
    occ = pres = None
    if need_pen:
        occ = torch.zeros(B, vocab, device=device)
        pres = torch.zeros(B, vocab, device=device)
        for i, r in enumerate(reqs):
            for tk in r.output_token_ids:
                occ[i, tk] += 1.0
            for tk in r.prompt_token_ids:
                pres[i, tk] = 1.0

    bias = None
    if any(p.logit_bias for p in sp):
        bias = torch.zeros(B, vocab, device=device)
        for i, p in enumerate(sp):
            for tk, b in (p.logit_bias or {}).items():
                bias[i, tk] += b

    disallowed = None
    for i, (p, r) in enumerate(zip(sp, reqs)):
        forbid = []
        if p.allowed_token_ids is not None:
            allowed = set(p.allowed_token_ids)
            forbid.extend(tk for tk in range(vocab) if tk not in allowed)
        for grp in (p.bad_words_token_ids or []):
            if len(grp) == 1:
                forbid.append(grp[0])
        # min_tokens / ignore_eos: block stop tokens until min_tokens reached
        if len(r.output_token_ids) < p.min_tokens or p.ignore_eos:
            forbid.extend(p.stop_token_ids)
        if forbid:
            if disallowed is None:
                disallowed = torch.zeros(B, vocab, dtype=torch.bool, device=device)
            disallowed[i, torch.tensor(sorted(set(forbid)), device=device)] = True

    # Host-side guard flags: known from the Python SamplingParams without any
    # device access, so sample_batched can branch on them without a
    # device->host sync (critical for the async NPU submit()/collect() split).
    has_top_k = any(p.top_k and p.top_k > 0 for p in sp)
    has_top_p = any(p.top_p < 1.0 for p in sp)
    has_min_p = any(p.min_p > 0.0 for p in sp)
    all_greedy_unprocessed = (
        all(p.temperature <= 0.0 for p in sp)
        and not need_pen
        and bias is None
        and disallowed is None
    )

    return SamplingTensors(temperature, top_k, top_p, min_p, presence, frequency,
                           repetition, occ, pres, bias, disallowed,
                           has_top_k=has_top_k, has_top_p=has_top_p,
                           has_min_p=has_min_p,
                           all_greedy_unprocessed=all_greedy_unprocessed), order
