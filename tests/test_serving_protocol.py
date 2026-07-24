import pytest

from auto_infer.serving.config import ServingConfig
from auto_infer.serving.protocol import (ChatCompletionRequest,
                                         CompletionRequest, error_payload,
                                         sampling_params)


def test_serving_defaults_and_derived_waiting_limit():
    cfg = ServingConfig(max_num_seqs=8)

    assert cfg.max_waiting_requests == 16
    assert cfg.tokenizer_batch_size == 32
    assert cfg.tokenizer_wait_ms == 2.0
    assert cfg.admission_wait_ms == 10.0
    assert cfg.sse_coalesce_ms == 5.0
    assert cfg.shutdown_grace_s == 30.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_num_seqs", 0),
        ("max_http_inflight", 0),
        ("max_waiting_requests", 0),
        ("max_waiting_tokens", 0),
        ("tokenizer_batch_size", 0),
        ("tokenizer_queue_capacity", 0),
        ("tokenizer_wait_ms", -1),
        ("admission_wait_ms", -1),
        ("sse_coalesce_ms", -1),
        ("shutdown_grace_s", -1),
    ],
)
def test_serving_config_rejects_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        ServingConfig(**{field: value})


def test_completion_rejects_invalid_sampling():
    with pytest.raises(ValueError):
        CompletionRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValueError, match="min_tokens"):
        CompletionRequest(prompt="x", max_tokens=1, min_tokens=2)


def test_protocol_maps_supported_sampling_fields():
    request = CompletionRequest(
        prompt="x",
        max_tokens=7,
        temperature=0.3,
        top_p=0.9,
        top_k=10,
        min_p=0.1,
        presence_penalty=0.2,
        frequency_penalty=0.4,
        repetition_penalty=1.1,
        min_tokens=2,
        ignore_eos=True,
        stop_token_ids=[4],
        seed=19,
    )

    params = sampling_params(request, eos_token_id=2)

    assert params.max_tokens == 7
    assert params.temperature == 0.3
    assert params.top_p == 0.9
    assert params.top_k == 10
    assert params.min_p == 0.1
    assert params.presence_penalty == 0.2
    assert params.frequency_penalty == 0.4
    assert params.repetition_penalty == 1.1
    assert params.min_tokens == 2
    assert params.ignore_eos is True
    assert params.eos_token_id == 2
    assert params.stop_token_ids == [4]
    assert params.seed == 19


def test_protocol_accepts_one_text_prompt_and_chat_messages():
    completion = CompletionRequest(prompt="hello")
    chat = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}]
    )

    assert completion.prompt == "hello"
    assert chat.messages[0].role == "user"
    assert chat.messages[0].content == "hello"


def test_protocol_rejects_unknown_fields():
    with pytest.raises(ValueError):
        CompletionRequest(prompt="hello", unsupported=True)


def test_error_payload_has_one_stable_openai_shaped_envelope():
    assert error_payload("queue full", error_type="overloaded", code="queue_full") == {
        "error": {
            "message": "queue full",
            "type": "overloaded",
            "param": None,
            "code": "queue_full",
        }
    }
