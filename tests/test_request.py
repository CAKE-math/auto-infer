import pytest

from auto_infer.engine.request import Request, SamplingParams, RequestStatus


def test_request_and_sampling_validation():
    with pytest.raises(ValueError, match="request_id"):
        Request("", [1], SamplingParams())
    with pytest.raises(ValueError, match="prompt"):
        Request("r", [], SamplingParams())
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=1.1)
    with pytest.raises(ValueError, match="min_tokens"):
        SamplingParams(max_tokens=2, min_tokens=3)


def make(**kw):
    return Request(request_id="r1", prompt_token_ids=[1, 2, 3],
                   sampling=SamplingParams(**kw))


def test_counts_and_append():
    r = make(max_tokens=2)
    assert r.num_prompt_tokens == 3
    assert r.num_tokens == 3
    assert r.status is RequestStatus.WAITING
    r.append_output_token(9)
    assert r.output_token_ids == [9]
    assert r.num_tokens == 4
    assert r.all_token_ids == [1, 2, 3, 9]


def test_finish_on_max_tokens():
    r = make(max_tokens=2)
    r.append_output_token(9)
    assert r.is_finished() is False
    r.append_output_token(10)
    assert r.is_finished() is True


def test_finish_on_stop_token():
    r = make(max_tokens=10, stop_token_ids=[7])
    r.append_output_token(7)
    assert r.is_finished() is True


def test_sampling_params_defaults_are_inert():
    p = SamplingParams()
    assert p.temperature == 0.0          # greedy
    assert p.top_k == 0 and p.top_p == 1.0 and p.min_p == 0.0
    assert p.presence_penalty == 0.0 and p.frequency_penalty == 0.0
    assert p.repetition_penalty == 1.0
    assert p.logit_bias is None and p.bad_words_token_ids is None
    assert p.allowed_token_ids is None
    assert p.min_tokens == 0 and p.ignore_eos is False


def test_sampling_params_accepts_full_surface():
    p = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05,
                       presence_penalty=0.5, frequency_penalty=0.2,
                       repetition_penalty=1.1, logit_bias={5: -10.0},
                       bad_words_token_ids=[[7]], allowed_token_ids=[1, 2, 3],
                       min_tokens=4, ignore_eos=True)
    assert p.logit_bias[5] == -10.0 and p.allowed_token_ids == [1, 2, 3]


def test_num_prefill_tokens_defaults_to_prompt_len():
    r = Request(request_id="a", prompt_token_ids=[1, 2, 3],
                sampling=SamplingParams())
    assert r.num_prefill_tokens == 3
