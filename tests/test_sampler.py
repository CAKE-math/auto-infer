import torch
import pytest

from auto_infer.layers.sampler import greedy, sample
from auto_infer.layers.sampler import SamplingTensors, sample_batched


def _greedy_tensors(B, vocab):
    return SamplingTensors(
        temperature=torch.zeros(B), top_k=torch.zeros(B, dtype=torch.long),
        top_p=torch.ones(B), min_p=torch.zeros(B),
        presence=torch.zeros(B), frequency=torch.zeros(B), repetition=torch.ones(B),
        occurrence_counts=None, prompt_presence=None, bias=None, disallowed_mask=None)


def test_batched_greedy_matches_per_row():
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 0.0]])
    out = sample_batched(logits, _greedy_tensors(2, 3))
    assert out.tolist() == [1, 0]


def test_all_greedy_unprocessed_skips_softmax_and_multinomial(monkeypatch):
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 2.0]])
    tensors = _greedy_tensors(2, 3)
    tensors.all_greedy_unprocessed = True
    monkeypatch.setattr(torch, "softmax", lambda *a, **k: pytest.fail("softmax called"))
    monkeypatch.setattr(
        torch, "multinomial", lambda *a, **k: pytest.fail("multinomial called"))

    assert sample_batched(logits, tensors).tolist() == [1, 0]


def test_batched_disallowed_mask_forbids_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    t = _greedy_tensors(1, 3)
    t.disallowed_mask = torch.tensor([[True, False, False]])   # forbid the argmax
    assert sample_batched(logits, t).tolist() == [1]


def test_batched_repetition_penalty_demotes_seen_token():
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    t = _greedy_tensors(1, 3)
    t.repetition = torch.tensor([2.0])
    t.occurrence_counts = torch.tensor([[1.0, 0.0, 0.0]])       # token 0 already seen
    t.prompt_presence = torch.zeros(1, 3)
    # token 0 (positive logit) divided by 2 -> 1.0 < 1.9 -> token 1 wins
    assert sample_batched(logits, t).tolist() == [1]


def test_batched_logit_bias_added():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    t = _greedy_tensors(1, 3)
    t.bias = torch.tensor([[0.0, 0.0, 5.0]])
    assert sample_batched(logits, t).tolist() == [2]


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 0.0]])
    assert greedy(logits).tolist() == [1, 0]


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert sample(logits, temperature=0.0).tolist() == [1]


def test_top_k_1_is_greedy():
    logits = torch.tensor([[0.1, 5.0, 0.2, 4.9]])
    g = torch.Generator().manual_seed(0)
    assert sample(logits, temperature=1.0, top_k=1, generator=g).tolist() == [1]


def test_top_p_restricts_support():
    # token 0 dominates; top_p small -> must pick token 0
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    out = [int(sample(logits, temperature=1.0, top_p=0.5, generator=g)[0]) for _ in range(10)]
    assert set(out) == {0}


def test_batched_matches_scalar_top_k_top_p():
    # Single-row scalar-vs-batched equivalence: both paths do
    # sort -> softmax -> cumsum -> multinomial(num_samples=1) on identically
    # shaped (1, vocab) tensors, so identically-seeded generators must draw
    # the same sample if the underlying math (post top_k/top_p masking)
    # agrees. Verified empirically that the two paths stay in lockstep here.
    logits = torch.tensor([[3.0, 1.0, 2.0, 0.5, 4.0, 0.1, 2.5, 1.5]])
    g1 = torch.Generator().manual_seed(42)
    scalar_out = sample(logits, temperature=0.8, top_k=3, top_p=0.9, generator=g1)

    vocab = logits.shape[-1]
    t = _greedy_tensors(1, vocab)
    t.temperature = torch.tensor([0.8])
    t.top_k = torch.tensor([3])
    t.top_p = torch.tensor([0.9])
    g2 = torch.Generator().manual_seed(42)
    batched_out = sample_batched(logits, t, generator=g2)

    assert scalar_out.tolist() == batched_out.tolist()


def test_batched_heterogeneous_top_k():
    # Two rows, top_k=1 each -> each row collapses to a single non--inf
    # entry, so multinomial is deterministic per row. Validates that the
    # per-row gather in the vectorized top_k path has no cross-talk.
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0], [1.0, 1.0, 10.0, 1.0]])
    t = _greedy_tensors(2, 4)
    t.temperature = torch.tensor([1.0, 1.0])
    t.top_k = torch.tensor([1, 1])
    assert sample_batched(logits, t).tolist() == [0, 2]


def test_batched_min_p_singleton():
    # Dominant token's softmax prob ~1.0; min_p=0.5 prunes every other
    # token below 0.5 * top -> singleton surviving set -> deterministic.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    t = _greedy_tensors(1, 4)
    t.temperature = torch.tensor([1.0])
    t.min_p = torch.tensor([0.5])
    assert sample_batched(logits, t).tolist() == [0]
