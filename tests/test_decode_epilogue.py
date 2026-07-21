from types import SimpleNamespace

import torch

from auto_infer.models.base import BaseCausalLM
from auto_infer.worker.decode_epilogue import is_capturable_greedy


def _request(**overrides):
    defaults = dict(
        temperature=0.0, presence_penalty=0.0, frequency_penalty=0.0,
        repetition_penalty=1.0, logit_bias=None, allowed_token_ids=None,
        bad_words_token_ids=None, min_tokens=0, ignore_eos=False,
        stop_token_ids=(),
    )
    defaults.update(overrides)
    return SimpleNamespace(
        sampling=SimpleNamespace(**defaults), output_token_ids=())


def test_only_unprocessed_greedy_batches_use_captured_epilogue():
    assert is_capturable_greedy([_request(), _request()])
    assert not is_capturable_greedy([_request(temperature=0.8)])
    assert not is_capturable_greedy([_request(logit_bias={7: 1.0})])
    assert not is_capturable_greedy([
        _request(presence_penalty=0.1)])


def test_token_constraints_require_external_sampler():
    assert not is_capturable_greedy([
        _request(allowed_token_ids=(1, 2))])
    assert not is_capturable_greedy([
        _request(bad_words_token_ids=((3,),))])
    assert not is_capturable_greedy([
        _request(ignore_eos=True, stop_token_ids=(9,))])
    assert not is_capturable_greedy([
        _request(min_tokens=4, stop_token_ids=(9,))])


def test_inactive_stop_constraints_do_not_block_capture():
    assert is_capturable_greedy([
        _request(min_tokens=0, ignore_eos=False, stop_token_ids=(9,))])


def test_logits_use_resident_weight_dtype_without_caching_fp32_head():
    model = BaseCausalLM()
    model.w = {"lm_head.weight": torch.randn(11, 5, dtype=torch.bfloat16)}
    hidden = torch.randn(3, 5, dtype=torch.bfloat16)
    output = torch.empty(3, 11, dtype=torch.bfloat16)
    address = output.data_ptr()

    returned = model.logits(hidden, out=output)

    assert returned.data_ptr() == address
    assert output.dtype == torch.bfloat16
    assert not hasattr(model, "_lm_head_fp32")
    torch.testing.assert_close(output, hidden @ model.w["lm_head.weight"].t())


def test_fp32_reference_logits_are_transient_and_do_not_change_default_policy():
    model = BaseCausalLM()
    model.w = {"lm_head.weight": torch.randn(11, 5, dtype=torch.bfloat16)}
    hidden = torch.randn(3, 5, dtype=torch.bfloat16)

    reference = model.logits(hidden, precision="float32")
    fast = model.logits(hidden)

    assert reference.dtype == torch.float32
    assert fast.dtype == torch.bfloat16
    assert not hasattr(model, "_lm_head_fp32")
    torch.testing.assert_close(
        reference, hidden.float() @ model.w["lm_head.weight"].float().t())
