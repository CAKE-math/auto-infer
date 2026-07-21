"""Distributed sharded loader (spec §15.1) — host test, no NPU. Builds a synthetic
multi-shard safetensors model with MoE-style names + index.json, then verifies:
 (1) build_weight_index maps every tensor to its shard file (index.json path);
 (2) full load (ep_size=1) reads every tensor, values intact;
 (3) EP-sharded load reads ONLY this rank's experts + all non-expert weights —
     non-local experts are never materialized (the 671B OOM-avoidance guarantee);
 (4) the union of all EP ranks' loads == the full set (complete, disjoint experts).
"""
import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from auto_infer.models.loader import (build_weight_index, expert_shard_predicate,
                                       load_sharded)

N_LAYERS, N_EXPERTS = 2, 8


def _make_model(d):
    """Write a 2-shard synthetic model: non-expert weights in shard 1, experts in
    shard 2, plus model.safetensors.index.json mapping names->shards."""
    shard1, shard2 = {}, {}
    weight_map = {}
    for i in range(N_LAYERS):
        for nm in ("input_layernorm.weight", "self_attn.q_proj.weight",
                   "mlp.gate.weight"):
            key = f"model.layers.{i}.{nm}"
            shard1[key] = torch.randn(4, 4)
            weight_map[key] = "model-00001-of-00002.safetensors"
        for e in range(N_EXPERTS):
            key = f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"
            shard2[key] = torch.full((4, 4), float(e))   # value = expert id (checkable)
            weight_map[key] = "model-00002-of-00002.safetensors"
    shard1["model.embed_tokens.weight"] = torch.randn(8, 4)
    weight_map["model.embed_tokens.weight"] = "model-00001-of-00002.safetensors"
    save_file(shard1, os.path.join(d, "model-00001-of-00002.safetensors"))
    save_file(shard2, os.path.join(d, "model-00002-of-00002.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def test_index_and_full_load():
    with tempfile.TemporaryDirectory() as d:
        _make_model(d)
        idx = build_weight_index(d)
        # every expert + non-expert name mapped to a shard file
        assert idx["model.layers.0.mlp.experts.7.gate_proj.weight"].endswith("00002-of-00002.safetensors")
        assert idx["model.embed_tokens.weight"].endswith("00001-of-00002.safetensors")
        full = load_sharded(d, lambda n: True)
        assert len(full) == len(idx)                      # loads everything
        # value integrity: expert e's tensor is all-e
        assert torch.equal(full["model.layers.1.mlp.experts.5.gate_proj.weight"],
                           torch.full((4, 4), 5.0))


def test_ep_shard_loads_only_local_experts():
    with tempfile.TemporaryDirectory() as d:
        _make_model(d)
        ep_size = 4                                       # 8 experts / 4 ranks = 2 each
        seen_experts = set()
        for ep_rank in range(ep_size):
            pred = expert_shard_predicate(N_EXPERTS, ep_size, ep_rank)
            w = load_sharded(d, pred)
            # non-expert weights present on every rank
            assert "model.embed_tokens.weight" in w
            assert "model.layers.0.self_attn.q_proj.weight" in w
            # only this rank's 2 experts loaded, per layer
            local = sorted({int(k.split(".experts.")[1].split(".")[0])
                            for k in w if ".experts." in k})
            assert local == [ep_rank * 2, ep_rank * 2 + 1], (ep_rank, local)
            # non-local experts NEVER materialized (the OOM-avoidance guarantee)
            for e in range(N_EXPERTS):
                key = f"model.layers.0.mlp.experts.{e}.gate_proj.weight"
                assert (key in w) == (e in (ep_rank * 2, ep_rank * 2 + 1))
            seen_experts.update(local)
        # union across ranks == all experts (complete + disjoint cover)
        assert seen_experts == set(range(N_EXPERTS))


if __name__ == "__main__":
    test_index_and_full_load()
    test_ep_shard_loads_only_local_experts()
    print("ALL PASS: sharded loader — index map, full load, EP-local expert sharding")
