"""Host-only tests for the weight-load prefetch + parallel-shard-read work
(feat/weight-load-prefetch). No NPU needed — CPU float32/bfloat16 tensors.

Covers:
 (a) `load_sharded` sequential (max_workers=1) vs parallel (max_workers=8)
     return value-identical dicts, correct device/dtype, for both a `wanted`
     predicate and `wanted=all` — proves the new ThreadPoolExecutor-per-file
     path (mirrors vLLM's `multi_thread_safetensors_weights_iterator`,
     weight_utils.py ~930) doesn't change what gets loaded.
 (b) `start_prefetch` is non-blocking (returns immediately) and never raises,
     both for a real checkpoint dir and for a missing path (mirrors vLLM's
     `maybe_prefetch_checkpoint`/`_prefetch_all_checkpoints`, ~800).
 (c) A tiny Qwen2 model built via the new `Qwen2Model.from_pretrained` (now
     streaming + prefetch) has IDENTICAL weights to the same checkpoint read
     the OLD way (`safetensors.torch.load_file` + `.to(device, dtype)`) —
     proves the Qwen2 migration in `models/qwen2.py` is behavior-preserving.
"""
import glob
import json
import os
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file

from auto_infer.models.loader import load_sharded, start_prefetch
from auto_infer.models.qwen2 import Qwen2Model, pack_qwen2_projections

HIDDEN, HEADS, KV_HEADS, INTER, VOCAB, LAYERS = 8, 2, 1, 16, 12, 2
HEAD_DIM = HIDDEN // HEADS


def _make_multi_shard_model(d, n_shards=3):
    """Write a multi-file safetensors checkpoint (no index.json, so
    build_weight_index falls back to header-scan) with distinguishable
    per-tensor values so equality checks are meaningful."""
    names_per_shard = [[] for _ in range(n_shards)]
    all_tensors = {}
    i = 0
    for layer in range(4):
        for nm in ("attn.weight", "mlp.weight", "norm.weight"):
            name = f"model.layers.{layer}.{nm}"
            names_per_shard[i % n_shards].append(name)
            all_tensors[name] = torch.arange(12, dtype=torch.float32).reshape(3, 4) + i
            i += 1
    for shard_idx, names in enumerate(names_per_shard):
        shard = {n: all_tensors[n] for n in names}
        save_file(shard, os.path.join(d, f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"))
    return all_tensors


def _write_tiny_qwen2(d, *, include_lm_head=False, tie_word_embeddings=True):
    cfg = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "intermediate_size": INTER, "vocab_size": VOCAB,
        "architectures": ["Qwen2ForCausalLM"], "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0, "tie_word_embeddings": tie_word_embeddings,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    torch.manual_seed(7)
    w = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
         "model.norm.weight": torch.randn(HIDDEN)}
    if include_lm_head:
        w["lm_head.weight"] = torch.randn(VOCAB, HIDDEN)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = torch.randn(HIDDEN)
        w[p + "post_attention_layernorm.weight"] = torch.randn(HIDDEN)
        w[p + "self_attn.q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.q_proj.bias"] = torch.randn(HEADS * HEAD_DIM)
        w[p + "self_attn.k_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.k_proj.bias"] = torch.randn(KV_HEADS * HEAD_DIM)
        w[p + "self_attn.v_proj.weight"] = torch.randn(KV_HEADS * HEAD_DIM, HIDDEN)
        w[p + "self_attn.v_proj.bias"] = torch.randn(KV_HEADS * HEAD_DIM)
        w[p + "self_attn.o_proj.weight"] = torch.randn(HIDDEN, HEADS * HEAD_DIM)
        w[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN)
        w[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN)
        w[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER)
    # split across 2 shard files like a real multi-shard HF checkpoint
    keys = list(w.keys())
    half = len(keys) // 2
    save_file({k: w[k] for k in keys[:half]}, os.path.join(d, "model-00001-of-00002.safetensors"))
    save_file({k: w[k] for k in keys[half:]}, os.path.join(d, "model-00002-of-00002.safetensors"))
    return w


def test_load_sharded_sequential_matches_parallel_full():
    with tempfile.TemporaryDirectory() as d:
        ref = _make_multi_shard_model(d)
        seq = load_sharded(d, lambda n: True, device="cpu", dtype=torch.float32, max_workers=1)
        par = load_sharded(d, lambda n: True, device="cpu", dtype=torch.float32, max_workers=8)
        assert set(seq) == set(ref) == set(par)
        for k in ref:
            assert seq[k].device.type == "cpu"
            assert seq[k].dtype == torch.float32
            assert torch.equal(seq[k], ref[k])
            assert torch.equal(par[k], seq[k])   # parallel == sequential, value-identical


def test_load_sharded_sequential_matches_parallel_with_predicate():
    with tempfile.TemporaryDirectory() as d:
        ref = _make_multi_shard_model(d)
        wanted = lambda n: "attn" in n  # noqa: E731 - subset predicate (TP/EP-shard style)
        seq = load_sharded(d, wanted, device="cpu", dtype=torch.bfloat16, max_workers=1)
        par = load_sharded(d, wanted, device="cpu", dtype=torch.bfloat16, max_workers=8)
        expected_names = {n for n in ref if wanted(n)}
        assert set(seq) == set(par) == expected_names
        assert expected_names and expected_names != set(ref)  # predicate actually filters
        for k in expected_names:
            assert seq[k].dtype == torch.bfloat16
            assert torch.equal(seq[k], par[k])
            assert torch.equal(seq[k].float(), ref[k].to(torch.bfloat16).float())


def test_start_prefetch_is_nonblocking_and_never_raises():
    with tempfile.TemporaryDirectory() as d:
        _make_multi_shard_model(d)
        start = time.perf_counter()
        thread = start_prefetch(d)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0   # must return immediately, not block on I/O
        thread.join(timeout=5)  # let the background warm finish before dir cleanup
        assert not thread.is_alive()


def test_start_prefetch_missing_path_never_raises():
    missing = "/nonexistent/path/for/prefetch/test/zzz"
    assert not glob.glob(os.path.join(missing, "*.safetensors"))
    thread = start_prefetch(missing)   # must not raise
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_qwen2_new_loader_matches_old():
    """The new Qwen2Model.from_pretrained (start_prefetch + load_sharded,
    streaming) must produce IDENTICAL weights to the OLD eager
    safetensors.torch.load_file(...) + .to(device, dtype) path it replaced."""
    with tempfile.TemporaryDirectory() as d:
        _write_tiny_qwen2(d)
        device, dtype = torch.device("cpu"), torch.float32

        # OLD path, reproduced verbatim from the pre-migration code.
        sd: dict[str, torch.Tensor] = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            sd.update(load_file(f))
        old_w = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        if "lm_head.weight" not in old_w:
            old_w["lm_head.weight"] = old_w["model.embed_tokens.weight"]
        pack_qwen2_projections(old_w, LAYERS)

        # NEW path: the real from_pretrained (prefetch + streaming loader).
        model = Qwen2Model.from_pretrained(d, device=device, dtype=dtype)

        assert set(model.w) == set(old_w)
        for k in old_w:
            assert model.w[k].dtype == old_w[k].dtype
            assert model.w[k].device.type == old_w[k].device.type
            assert torch.equal(model.w[k], old_w[k]), f"mismatch at {k}"
        # tied embeddings still resolved identically
        assert torch.equal(model.w["lm_head.weight"], model.w["model.embed_tokens.weight"])


def test_tied_checkpoint_head_aliases_embedding_even_when_both_are_stored():
    with tempfile.TemporaryDirectory() as d:
        original = _write_tiny_qwen2(d, include_lm_head=True,
                                     tie_word_embeddings=True)
        assert not torch.equal(
            original["lm_head.weight"], original["model.embed_tokens.weight"])

        model = Qwen2Model.from_pretrained(
            d, device=torch.device("cpu"), dtype=torch.float32)

        assert model.w["lm_head.weight"].data_ptr() == \
            model.w["model.embed_tokens.weight"].data_ptr()


def test_untied_checkpoint_preserves_independent_language_model_head():
    with tempfile.TemporaryDirectory() as d:
        _write_tiny_qwen2(d, include_lm_head=True, tie_word_embeddings=False)

        model = Qwen2Model.from_pretrained(
            d, device=torch.device("cpu"), dtype=torch.float32)

        assert model.w["lm_head.weight"].data_ptr() != \
            model.w["model.embed_tokens.weight"].data_ptr()


if __name__ == "__main__":
    test_load_sharded_sequential_matches_parallel_full()
    test_load_sharded_sequential_matches_parallel_with_predicate()
    test_start_prefetch_is_nonblocking_and_never_raises()
    test_start_prefetch_missing_path_never_raises()
    test_qwen2_new_loader_matches_old()
    print("ALL PASS: weight loader — prefetch, parallel==sequential, Qwen2 migration parity")
