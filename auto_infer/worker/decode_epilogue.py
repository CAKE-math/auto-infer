"""Host-known selection between captured greedy and general decode sampling."""


def is_capturable_greedy(requests) -> bool:
    """Whether requests may use the captured FP32 lm-head + argmax path."""
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
