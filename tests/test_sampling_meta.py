import pytest
import torch
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.layers.sampling_meta import build_sampling_tensors


def _req(rid, temp=0.0, outs=(), bad=None, presence_penalty=0.0, frequency_penalty=0.0):
    r = Request(request_id=rid, prompt_token_ids=[1, 2],
                sampling=SamplingParams(temperature=temp, bad_words_token_ids=bad,
                                         presence_penalty=presence_penalty,
                                         frequency_penalty=frequency_penalty))
    for tk in outs:
        r.append_output_token(tk)
    return r


def test_order_and_temperature_row_alignment():
    reqs = [_req("a", temp=0.0), _req("b", temp=0.7)]
    t, order = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert order == ["a", "b"]
    # float32 storage: 0.7 is not exactly representable, so compare with tolerance
    assert t.temperature.tolist() == pytest.approx([0.0, 0.7])


def test_occurrence_counts_from_outputs():
    # need_pen gate requires an active penalty for occurrence_counts /
    # prompt_presence to be built at all.
    reqs = [_req("a", outs=(3, 3, 5), presence_penalty=0.1)]
    t, _ = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert t.occurrence_counts is not None
    assert t.occurrence_counts[0, 3].item() == 2.0
    assert t.occurrence_counts[0, 5].item() == 1.0


def test_no_penalty_skips_occurrence_counts():
    # Default SamplingParams (no presence/frequency/repetition penalty) must
    # not pay for building the (B, vocab) occurrence_counts / prompt_presence
    # tensors on every step.
    reqs = [_req("a", outs=(3, 3, 5))]
    t, _ = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert t.occurrence_counts is None
    assert t.prompt_presence is None
    assert t.all_greedy_unprocessed is True


def test_bad_words_sets_disallowed_mask():
    reqs = [_req("a", bad=[[7]])]
    t, _ = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert bool(t.disallowed_mask[0, 7]) is True
    assert t.all_greedy_unprocessed is False


@pytest.mark.parametrize("params", [
    SamplingParams(temperature=0.5),
    SamplingParams(logit_bias={3: 1.0}),
    SamplingParams(presence_penalty=0.1),
    SamplingParams(frequency_penalty=0.1),
    SamplingParams(repetition_penalty=1.1),
    SamplingParams(allowed_token_ids=[1, 2]),
])
def test_processed_or_random_sampling_is_not_plain_greedy(params):
    req = Request(request_id="a", prompt_token_ids=[1, 2], sampling=params)
    tensors, _ = build_sampling_tensors([req], vocab=8, device=torch.device("cpu"))
    assert tensors.all_greedy_unprocessed is False
