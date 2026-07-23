"""Host-only unit tests for worker/graph_decode_runner.py (SP3 task; updated
for SP5's generalized `attention(layer_idx, x, ctx)` seam; SP6 adds
`GraphMlaBackend` coverage + model-agnostic registry selection).

The graph/FIA/scatter ops (`npu_scatter_pa_kv_cache`, FIA-v2 `.out`,
`graph_task_group_*`, `NPUGraph`, and DeepSeek's MoE grouped-GEMM ops) are
NPU-only and cannot run here — correctness of `GraphGqaBackend`/
`GraphMlaBackend`'s `_write_kv/_attn/update` and `GraphPagedRunner._capture/
_graph` end-to-end is gated on npu2 (`scripts/verify_qwen2_graphdecode_batched.py`,
`scripts/smoke_graph_engine.py`, `scripts/verify_deepseek_graphdecode.py`).
This file covers the PURE-PYTHON slice that was extracted specifically to be
host-testable: gear selection and host-side batch marshaling (no device/NPU
ops), `GraphGqaBackend`/`GraphMlaBackend`'s construction/allocation surface
(CPU tensors only, like `GqaFIABackend`'s tests in test_attention_backend.py)
and their capture-mode bookkeeping, plus `Qwen2Model`/`DeepseekV2Model`'s
attention-registry model-agnostic backend selection."""
import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

import auto_infer.worker.graph_decode_runner as graph_runner
from auto_infer.engine.execution import DeviceTokenBatch
from auto_infer.layers.attention.gqa import GraphGqaBackend
from auto_infer.layers.attention.mla import GraphMlaBackend
from auto_infer.worker.graph_decode_runner import (
    GEARS,
    GraphPagedRunner,
    _PrefillGear,
    _Gear,
    _marshal_prefill_batch,
    _gather_sample_hidden,
    _scratch_blocks_for_gears,
    _select_prefill_gear,
    _select_gear,
)


# ---------------------------------------------------------------------------
# _select_gear
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("B,max_gear,expected", [
    (1, 64, 1),
    (2, 64, 2),
    (3, 64, 4),
    (4, 64, 4),
    (5, 64, 8),
    (64, 64, 64),
    (9, 16, 16),
])
def test_select_gear_picks_smallest_covering_gear(B, max_gear, expected):
    assert _select_gear(B, max_gear) == expected


def test_select_gear_returns_none_when_batch_exceeds_max_gear():
    assert _select_gear(100, 64) is None
    assert _select_gear(5, 4) is None       # next gear (8) exceeds cap


def test_select_gear_respects_max_gear_cap_below_largest_gear():
    # B=32 would normally hit gear 32, but max_gear=16 caps it out entirely
    assert _select_gear(32, 16) is None
    assert _select_gear(16, 16) == 16


def test_gears_constant_sorted_ascending():
    assert GEARS == sorted(GEARS)


def test_prefill_capture_sizes_match_vllm_policy():
    assert graph_runner._prefill_capture_sizes(32) == [1, 2, 4, 8, 16, 24, 32]
    assert len(graph_runner._prefill_capture_sizes(256)) == 35


@pytest.mark.parametrize("tokens,expected", [
    (1, 1),
    (3, 4),
    (10, 16),
    (17, 24),
    (25, 32),
    (33, None),
])
def test_select_prefill_gear_uses_flattened_token_count(tokens, expected):
    assert _select_prefill_gear(tokens, max_gear=32) == expected


def test_prefill_gear_is_independent_from_decode_batch_cap():
    assert _select_gear(33, max_gear=32) is None
    assert _select_prefill_gear(144, max_gear=256) == 144


@pytest.mark.parametrize("max_gear,expected", [
    (1, 1), (3, 3), (16, 16), (32, 32), (64, 64),
])
def test_scratch_capacity_covers_every_accepted_prefill_shape(max_gear, expected):
    assert _scratch_blocks_for_gears(max_gear) == expected


def test_prefill_prewarm_attempts_each_token_gear_once():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner.max_gear = 32
    runner.max_prefill_tokens = 256
    runner.prefill_gears = {}
    runner.failed_prefill_gears = set()
    runner.stats = {"prefill_graph_capture_attempts": 0,
                    "prefill_graph_capture_failures": 0}
    attempted = []
    runner._capture_prefill = lambda gear: attempted.append(gear) or object()

    runner._prewarm_prefill_gears()

    assert attempted == graph_runner._prefill_capture_sizes(256)
    assert sorted(runner.prefill_gears) == attempted
    assert runner.stats["prefill_graph_capture_attempts"] == 35
    assert runner._prefill_prewarm_active is False


def test_prefill_capture_counts_runtime_attempts():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner._prefill_prewarm_active = False
    runner.stats = {"prefill_graph_online_captures": 0}
    runner.model = object()

    with pytest.raises(AttributeError):
        runner._capture_prefill(1)

    assert runner.stats["prefill_graph_online_captures"] == 1


def test_prefill_prewarm_isolates_failed_gear():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner.max_gear = 4
    runner.max_prefill_tokens = 4
    runner.prefill_gears = {}
    runner.failed_prefill_gears = set()
    runner.stats = {"prefill_graph_capture_attempts": 0,
                    "prefill_graph_capture_failures": 0}
    attempts = []

    def capture(gear):
        attempts.append(gear)
        if gear == 2:
            raise RuntimeError("gear cannot be captured")
        return object()

    runner._capture_prefill = capture
    runner._prewarm_prefill_gears()

    assert attempts == [1, 2, 4]
    assert sorted(runner.prefill_gears) == [1, 4]
    assert runner.failed_prefill_gears == {2}
    assert runner.stats["prefill_graph_capture_failures"] == 1


def test_runtime_prefill_lookup_never_captures():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner.prefill_gears = {16: object()}
    runner.failed_prefill_gears = {24}
    runner._capture_prefill = lambda gear: pytest.fail("online capture")

    assert runner._get_prefill_gear(16) is runner.prefill_gears[16]
    assert runner._get_prefill_gear(24) is None
    assert runner._get_prefill_gear(32) is None


# ---------------------------------------------------------------------------
# host-side batch marshaling stubs
# ---------------------------------------------------------------------------

class _FakeReq:
    def __init__(self, all_token_ids, num_computed_tokens, num_prefill_tokens=None):
        self.all_token_ids = all_token_ids
        self.num_computed_tokens = num_computed_tokens
        self.num_prefill_tokens = num_prefill_tokens if num_prefill_tokens is not None \
            else len(all_token_ids)


class _FakeSR:
    def __init__(self, request_id, num_tokens_to_compute):
        self.request_id = request_id
        self.num_tokens_to_compute = num_tokens_to_compute


class _FakeScheduler:
    def __init__(self, requests, block_tables):
        self._requests = requests
        self.block_tables = block_tables

    def get_request(self, rid):
        return self._requests[rid]


def test_marshal_prefill_batch_single_request():
    reqs = {"a": _FakeReq(all_token_ids=[10, 11, 12, 13, 14], num_computed_tokens=0)}
    sched = _FakeScheduler(reqs, {"a": [0, 1]})
    sr = [_FakeSR("a", 3)]
    flat_ids, flat_pos, slots, cu_q, kv_lens, bt_rows, sample_idx, maxb = \
        _marshal_prefill_batch(sr, sched, block_size=4)
    assert flat_ids == [10, 11, 12]
    assert flat_pos == [0, 1, 2]
    assert slots == [0, 1, 2]              # block 0 (bt[0//4]=bt[0]=0) * 4 + {0,1,2}
    assert cu_q == [3]
    assert kv_lens == [3]
    assert sample_idx == {"a": 2}          # qacc - 1
    assert bt_rows == [[0, 1]]
    assert maxb == 2


def test_marshal_prefill_batch_two_requests_cumulative_qlen():
    reqs = {
        "a": _FakeReq(all_token_ids=list(range(10)), num_computed_tokens=2),
        "b": _FakeReq(all_token_ids=list(range(10, 20)), num_computed_tokens=0),
    }
    sched = _FakeScheduler(reqs, {"a": [5], "b": [6, 7]})
    sr = [_FakeSR("a", 2), _FakeSR("b", 1)]
    flat_ids, flat_pos, slots, cu_q, kv_lens, bt_rows, sample_idx, maxb = \
        _marshal_prefill_batch(sr, sched, block_size=4)
    # request "a": positions 2,3 (start=2, n=2) -> ids [2,3]
    # request "b": position 0 -> id [10]
    assert flat_ids == [2, 3, 10]
    assert flat_pos == [2, 3, 0]
    assert cu_q == [2, 3]                  # cumulative: 2, then 2+1
    assert kv_lens == [4, 1]               # start+n per request: 2+2=4, 0+1=1
    assert sample_idx == {"a": 1, "b": 2}  # last row index of each request's chunk
    assert maxb == 2


def test_prefill_gathers_only_rows_that_produce_samples_before_lm_head():
    hidden = torch.arange(45, dtype=torch.float32).reshape(9, 5)

    selected = _gather_sample_hidden(hidden, [8])

    assert selected.shape == (1, 5)
    assert torch.equal(selected[0], hidden[8])


# ---------------------------------------------------------------------------
# _Gear static buffers (pure torch, CPU)
# ---------------------------------------------------------------------------

def test_gear_static_buffer_shapes_and_dtypes():
    gear = _Gear(g=4, max_blocks=6, vocab=32,
                 device=torch.device("cpu"), dtype=torch.float32)
    assert gear.tid.shape == (4,) and gear.tid.dtype == torch.long
    assert gear.ppos.shape == (4,) and gear.ppos.dtype == torch.long
    assert gear.pslot.shape == (4,) and gear.pslot.dtype == torch.int32
    assert gear.bt.shape == (4, 6) and gear.bt.dtype == torch.int32
    assert gear.logits.shape == (4, 32) and gear.logits.dtype == torch.float32
    assert not hasattr(gear, "hout")
    assert gear.sampled.shape == (4,) and gear.sampled.dtype == torch.long
    assert gear.active_token_mask.shape == (4,)
    assert gear.active_token_mask.dtype == torch.bool
    assert gear.qlen_cum == [1, 2, 3, 4]
    assert gear.reg == []
    assert gear.graph is None
    assert gear.pipeline is None
    assert gear.stager is None


def test_gear_logits_follow_model_dtype():
    gear = _Gear(g=4, max_blocks=6, vocab=32,
                 device=torch.device("cpu"), dtype=torch.bfloat16)
    assert gear.logits.dtype == torch.bfloat16


def test_prefill_gear_owns_fixed_query_and_sample_buffers():
    gear = _PrefillGear(
        8, max_blocks=6, vocab=32,
        device=torch.device("cpu"), dtype=torch.bfloat16)

    assert gear.token_ids.shape == (8,)
    assert gear.positions.shape == (8,)
    assert gear.slots.shape == (8,)
    assert gear.block_table.shape == (8, 6)
    assert gear.sample_rows.shape == (8,)
    assert gear.logits.shape == (8, 32)
    assert gear.sampled.shape == (8,)
    assert gear.active_token_mask.shape == (8,)
    assert gear.active_token_mask.dtype == torch.bool
    assert gear.logits.dtype == torch.bfloat16


def test_async_sampled_refs_use_stable_token_store_without_clone():
    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    static = torch.tensor([3, 5, 7])
    stable = torch.tensor([3, 5, 7, 0])
    token_batch = DeviceTokenBatch.from_rows(
        stable, ("a", "b", "c"), (0, 1, 2))

    batch = runner.sampled_of({
        "tokens": static, "order": ["a", "b", "c"],
        "token_batch": token_batch})
    static.fill_(99)

    assert batch.tokens.tolist() == [3, 5, 7, 0]
    assert batch.tokens.data_ptr() != static.data_ptr()


def test_async_spills_only_refs_still_owned_by_released_slot():
    from auto_infer.worker.async_slots import DeviceTokenStore

    runner = GraphPagedRunner.__new__(GraphPagedRunner)
    runner._token_store = DeviceTokenStore(4, torch.device("cpu"))
    old = DeviceTokenBatch.from_output(
        torch.tensor([3, 5]), ("a", "b"))
    newer = DeviceTokenBatch.from_output(torch.tensor([7]), ("a",))
    refs = old.refs()
    refs["a"] = newer.refs()["a"]

    replacements = runner.stabilize_refs(
        {"token_batch": old}, refs)

    assert set(replacements) == {"b"}
    assert int(replacements["b"].owner.tokens[replacements["b"].row]) == 5


# ---------------------------------------------------------------------------
# GraphGqaBackend — construction / allocation / capture toggle
# (write_kv/attn/update call straight into torch_npu ops and are exercised by
# the NPU-only verify scripts, not here — same split as GqaFIABackend.)
# ---------------------------------------------------------------------------

def test_graph_attention_backend_stores_construction_args():
    backend = GraphGqaBackend(n_q_heads=16, n_kv_heads=2, head_dim=64, scale=0.125,
                                    num_layers=3, device=torch.device("cpu"), dtype=torch.float32)
    assert (backend.n_q_heads, backend.n_kv_heads, backend.head_dim, backend.scale) == (
        16, 2, 64, 0.125)
    assert (backend.num_layers, backend.device, backend.dtype) == (
        3, torch.device("cpu"), torch.float32)
    assert backend.capturing is False
    assert backend.reg == []
    assert backend.NZ == 16


def test_graph_attention_backend_alloc_kv_caches_shape_dtype_device():
    n_kv, hd, num_layers = 4, 16, 3
    backend = GraphGqaBackend(n_q_heads=8, n_kv_heads=n_kv, head_dim=hd, scale=hd ** -0.5,
                                    num_layers=num_layers, device=torch.device("cpu"),
                                    dtype=torch.float32)
    caches = backend.alloc_kv_caches(num_blocks=10, block_size=32)
    assert len(caches) == num_layers
    for kc, vc in caches:
        assert kc.shape == vc.shape == (10, 32, n_kv, hd)
        assert kc.dtype == vc.dtype == torch.float32
        assert kc.device.type == "cpu"
        # k and v must be distinct tensors, not aliases
        assert kc.data_ptr() != vc.data_ptr()


def test_graph_attention_backend_begin_end_capture_toggle_and_reset_reg():
    backend = GraphGqaBackend(n_q_heads=2, n_kv_heads=1, head_dim=4, scale=1.0,
                                    num_layers=2, device=torch.device("cpu"), dtype=torch.float32)
    backend.reg.append(("stale-handle",))
    backend.begin_capture()
    assert backend.capturing is True
    assert backend.reg == []                # begin_capture resets to a fresh list
    backend.reg.append(("layer0-handle",))
    backend.end_capture()
    assert backend.capturing is False
    assert backend.reg == [("layer0-handle",)]    # end_capture does NOT clear reg


def test_graph_attention_backend_is_an_attention_backend_subclass():
    from auto_infer.layers.attention.base import AttentionBackend
    assert issubclass(GraphGqaBackend, AttentionBackend)


# ---------------------------------------------------------------------------
# GraphMlaBackend (SP6) — construction / allocation / capture toggle, same
# split as GraphGqaBackend above: `_write_kv`/`_attn`/`update` call straight
# into torch_npu ops and are exercised by the NPU-only verify scripts, not
# here.
# ---------------------------------------------------------------------------

def test_graph_mla_backend_stores_construction_args():
    backend = GraphMlaBackend({"a": 1}, num_heads=16, qk_nope=128, qk_rope=64,
                              v_head_dim=128, kv_lora_rank=512, q_lora_rank=None,
                              rms_eps=1e-6, softmax_scale=0.1, num_layers=3,
                              device=torch.device("cpu"), dtype=torch.float32)
    assert backend.w == {"a": 1}
    assert (backend.num_heads, backend.qk_nope, backend.qk_rope, backend.v_head_dim) == (
        16, 128, 64, 128)
    assert (backend.kv_lora_rank, backend.q_lora_rank, backend.rms_eps) == (512, None, 1e-6)
    assert backend.softmax_scale == 0.1
    assert (backend.num_layers, backend.device, backend.dtype) == (
        3, torch.device("cpu"), torch.float32)
    assert backend.capturing is False
    assert backend.reg == []
    assert backend.NZ == 16


def test_graph_mla_backend_advertises_prefill_graph_support():
    assert GraphMlaBackend.supports_prefill_graph is True


def test_graph_mla_backend_alloc_kv_caches_shape_dtype_device_non_absorbed():
    nh, nope, rope, vd, kvl, layers = 4, 128, 64, 128, 512, 3
    backend = GraphMlaBackend({}, num_heads=nh, qk_nope=nope, qk_rope=rope, v_head_dim=vd,
                              kv_lora_rank=kvl, q_lora_rank=None, rms_eps=1e-6,
                              softmax_scale=(nope + rope) ** -0.5, num_layers=layers,
                              device=torch.device("cpu"), dtype=torch.float32)
    caches = backend.alloc_kv_caches(num_blocks=5, block_size=16)
    assert len(caches) == layers
    for kc, vc in caches:
        assert kc.shape == (5, 16, nh, nope + rope)
        assert vc.shape == (5, 16, nh, vd)
        assert kc.dtype == vc.dtype == torch.float32
        assert kc.device.type == vc.device.type == "cpu"
        # k and v must be distinct tensors, not aliases (same contract as GraphGqaBackend)
        assert kc.data_ptr() != vc.data_ptr()


def test_graph_mla_backend_begin_end_capture_toggle_and_reset_reg():
    backend = GraphMlaBackend({}, num_heads=2, qk_nope=2, qk_rope=2, v_head_dim=2,
                              kv_lora_rank=4, q_lora_rank=None, rms_eps=1e-6,
                              softmax_scale=1.0, num_layers=2,
                              device=torch.device("cpu"), dtype=torch.float32)
    backend.reg.append(("stale-handle",))
    backend.begin_capture()
    assert backend.capturing is True
    assert backend.reg == []                # begin_capture resets to a fresh list
    backend.reg.append(("layer0-handle",))
    backend.end_capture()
    assert backend.capturing is False
    assert backend.reg == [("layer0-handle",)]    # end_capture does NOT clear reg


def test_graph_mla_backend_is_an_attention_backend_subclass():
    from auto_infer.layers.attention.base import AttentionBackend
    assert issubclass(GraphMlaBackend, AttentionBackend)


def test_graph_attention_backends_share_one_fia_lifecycle():
    from auto_infer.layers.attention.graph_fia import GraphFiaLifecycle

    for backend_type in (GraphGqaBackend, GraphMlaBackend):
        assert issubclass(backend_type, GraphFiaLifecycle)
        assert "begin_capture" not in backend_type.__dict__
        assert "end_capture" not in backend_type.__dict__
        assert "update" not in backend_type.__dict__


def test_graph_attention_kv_writers_resolve_the_npu_operator(monkeypatch):
    calls = []
    fake_npu = type("FakeNpu", (), {
        "npu_scatter_pa_kv_cache": staticmethod(
            lambda *args: calls.append(args))})
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", fake_npu)
    backends = (
        GraphGqaBackend(
            n_q_heads=1, n_kv_heads=1, head_dim=16, scale=1.0,
            num_layers=1, device=torch.device("cpu"), dtype=torch.float32),
        GraphMlaBackend(
            {}, num_heads=1, qk_nope=8, qk_rope=8, v_head_dim=16,
            kv_lora_rank=8, q_lora_rank=None, rms_eps=1e-6,
            softmax_scale=1.0, num_layers=1, device=torch.device("cpu"),
            dtype=torch.float32),
    )
    for backend in backends:
        ctx = type("Ctx", (), {
            "kv_caches": backend.alloc_kv_caches(1, 2),
            "slot_mapping": torch.tensor([0], dtype=torch.int32),
        })()
        backend._write_kv(
            0, torch.zeros(1, 1, 16), torch.zeros(1, 1, 16), ctx)

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# attention registry (SP6) — model-agnostic backend selection: Qwen2Model ->
# GraphGqaBackend, DeepseekV2Model -> GraphMlaBackend. Uses tiny real models
# (CPU, float32) loaded through each model's normal `from_pretrained`, since
# registry construction/cache allocation are pure-Python/tensor-alloc — no
# torch_npu ops involved (those live behind `attention`/`_write_kv`/`_attn`).
# ---------------------------------------------------------------------------

def test_qwen2_registry_returns_graph_gqa_backend_with_matching_caches():
    from auto_infer.models.qwen2 import Qwen2Model
    hidden, heads, kv_heads, inter, vocab, layers = 8, 2, 1, 16, 12, 2
    head_dim = hidden // heads
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "intermediate_size": inter, "vocab_size": vocab,
            "architectures": ["Qwen2ForCausalLM"], "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0, "tie_word_embeddings": True,
        }
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(0)
        w = {"model.embed_tokens.weight": torch.randn(vocab, hidden),
             "model.norm.weight": torch.randn(hidden) * 0.1 + 1.0}
        for i in range(layers):
            p = f"model.layers.{i}."
            w[p + "input_layernorm.weight"] = torch.randn(hidden) * 0.1 + 1.0
            w[p + "post_attention_layernorm.weight"] = torch.randn(hidden) * 0.1 + 1.0
            w[p + "self_attn.q_proj.weight"] = torch.randn(heads * head_dim, hidden) * 0.1
            w[p + "self_attn.q_proj.bias"] = torch.randn(heads * head_dim) * 0.1
            w[p + "self_attn.k_proj.weight"] = torch.randn(kv_heads * head_dim, hidden) * 0.1
            w[p + "self_attn.k_proj.bias"] = torch.randn(kv_heads * head_dim) * 0.1
            w[p + "self_attn.v_proj.weight"] = torch.randn(kv_heads * head_dim, hidden) * 0.1
            w[p + "self_attn.v_proj.bias"] = torch.randn(kv_heads * head_dim) * 0.1
            w[p + "self_attn.o_proj.weight"] = torch.randn(hidden, heads * head_dim) * 0.1
            w[p + "mlp.gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
            w[p + "mlp.up_proj.weight"] = torch.randn(inter, hidden) * 0.1
            w[p + "mlp.down_proj.weight"] = torch.randn(hidden, inter) * 0.1
        save_file(w, os.path.join(d, "model.safetensors"))

        model = Qwen2Model.from_pretrained(d, device=torch.device("cpu"), dtype=torch.float32)
        from auto_infer.layers.attention.registry import build_attention_backend
        backend, caches = build_attention_backend(model, "graph", num_blocks=5, block_size=16)
        assert isinstance(backend, GraphGqaBackend)
        assert len(caches) == layers
        for kc, vc in caches:
            assert kc.shape == vc.shape == (5, 16, kv_heads, head_dim)


def test_deepseek_registry_returns_graph_mla_backend_with_matching_caches():
    from auto_infer.models.deepseek_v2 import DeepseekV2Model
    hidden, heads, inter, vocab, layers = 8, 2, 8, 12, 2
    kv_lora, nope, rope, vd = 4, 2, 2, 2
    n_routed, n_shared, top_k, first_k_dense = 2, 1, 1, 1
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": heads,
            "kv_lora_rank": kv_lora, "q_lora_rank": None,
            "qk_nope_head_dim": nope, "qk_rope_head_dim": rope, "v_head_dim": vd,
            "vocab_size": vocab, "architectures": ["DeepseekV2ForCausalLM"],
            "rms_norm_eps": 1e-6, "rope_theta": 10000, "n_routed_experts": n_routed,
            "n_shared_experts": n_shared, "num_experts_per_tok": top_k,
            "first_k_dense_replace": first_k_dense, "norm_topk_prob": False,
            "scoring_func": "softmax", "topk_method": "greedy",
            "routed_scaling_factor": 1.0, "rope_scaling": None,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(1)
        qk = nope + rope
        w = {"model.embed_tokens.weight": torch.randn(vocab, hidden) * 0.1,
             "model.norm.weight": torch.randn(hidden) * 0.1 + 1.0}
        for i in range(layers):
            p = f"model.layers.{i}."
            w[p + "input_layernorm.weight"] = torch.randn(hidden) * 0.1 + 1.0
            w[p + "post_attention_layernorm.weight"] = torch.randn(hidden) * 0.1 + 1.0
            w[p + "self_attn.q_proj.weight"] = torch.randn(heads * qk, hidden) * 0.1
            w[p + "self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(kv_lora + rope, hidden) * 0.1
            w[p + "self_attn.kv_a_layernorm.weight"] = torch.randn(kv_lora) * 0.1 + 1.0
            w[p + "self_attn.kv_b_proj.weight"] = torch.randn(heads * (nope + vd), kv_lora) * 0.1
            w[p + "self_attn.o_proj.weight"] = torch.randn(hidden, heads * vd) * 0.1
            if i < first_k_dense:
                w[p + "mlp.gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
                w[p + "mlp.up_proj.weight"] = torch.randn(inter, hidden) * 0.1
                w[p + "mlp.down_proj.weight"] = torch.randn(hidden, inter) * 0.1
            else:
                mp = p + "mlp."
                w[mp + "gate.weight"] = torch.randn(n_routed, hidden) * 0.1
                for e in range(n_routed):
                    ep = f"{mp}experts.{e}."
                    w[ep + "gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
                    w[ep + "up_proj.weight"] = torch.randn(inter, hidden) * 0.1
                    w[ep + "down_proj.weight"] = torch.randn(hidden, inter) * 0.1
                sp = mp + "shared_experts."
                w[sp + "gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
                w[sp + "up_proj.weight"] = torch.randn(inter, hidden) * 0.1
                w[sp + "down_proj.weight"] = torch.randn(hidden, inter) * 0.1
        save_file(w, os.path.join(d, "model.safetensors"))

        model = DeepseekV2Model.from_pretrained(d, device=torch.device("cpu"), dtype=torch.float32)
        from auto_infer.layers.attention.registry import build_attention_backend
        backend, caches = build_attention_backend(model, "graph", num_blocks=5, block_size=16)
        assert isinstance(backend, GraphMlaBackend)
        assert len(caches) == layers
        for kc, vc in caches:
            assert kc.shape == (5, 16, heads, nope + rope)
            assert vc.shape == (5, 16, heads, vd)
        # non-absorbed regardless of the model's own `_mla_absorb` flag — the
        # graph path does not support absorbed MLA (block_size-128 constraint)
        model._mla_absorb = True
        backend2, caches2 = build_attention_backend(model, "graph", num_blocks=5, block_size=16)
        assert isinstance(backend2, GraphMlaBackend)
        for kc, vc in caches2:
            assert kc.shape == (5, 16, heads, nope + rope)   # still non-absorbed shape
