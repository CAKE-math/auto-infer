"""SP-MoE (spec §6; matches omni-npu/omni-models use_sequence_parallel_moe, used by
DeepSeek-V3) — host gloo test, no NPU. Launches N gloo ranks and verifies the
sequence-parallel token sharding is CORRECT:
 (1) sp_chunk pads tokens to a multiple of sp_size and hands each rank a disjoint
     1/sp shard; sp_all_gather reconstructs the exact original (padding dropped);
 (2) a PER-TOKEN function applied shard-wise then all-gathered == applied to the
     full sequence (the invariant that makes SP-MoE numerically identical to
     non-SP MoE, since MoE routing is independent per token).
Run standalone (spawns processes); not collected by the NPU suite.
"""
from auto_infer.deploy.launcher import LauncherConfig, launch


def worker(rank, world_size, role):
    import torch
    import torch.distributed as dist
    from auto_infer.distributed import parallel_state as ps

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # wire SP to share the (here: world) group, sp_size = world_size
    ps._TP_GROUP = dist.group.WORLD
    ps._TP_SIZE = world_size
    ps._TP_RANK = rank
    ps._SP_SIZE = world_size

    torch.manual_seed(0)                                  # same full tensor on every rank
    T, H = 13, 4                                          # 13 tokens, NOT divisible by world -> exercises padding
    x = torch.randn(T, H)

    # (1) chunk -> all_gather round-trip == original
    shard, num_tokens = ps.sp_chunk(x)
    assert num_tokens == T
    expected_shard_rows = ((T + world_size - 1) // world_size)
    assert shard.shape[0] == expected_shard_rows, (shard.shape, expected_shard_rows)
    gathered = ps.sp_all_gather(shard, num_tokens)
    roundtrip_ok = torch.equal(gathered, x)

    # (2) per-token function: shard-wise + gather == full
    W = torch.randn(H, H); torch.manual_seed(0)           # deterministic W across ranks
    W = torch.randn(H, H)
    def per_token(t):                                     # any row-independent op
        return torch.relu(t @ W)
    sp_result = ps.sp_all_gather(per_token(shard), num_tokens)
    full_result = per_token(x)
    equiv_ok = torch.allclose(sp_result, full_result, atol=1e-5)

    dist.destroy_process_group()
    return {"rank": rank, "roundtrip_ok": bool(roundtrip_ok), "equiv_ok": bool(equiv_ok),
            "shard_rows": shard.shape[0]}


if __name__ == "__main__":
    cfg = LauncherConfig(nnodes=1, nproc_per_node=4, master_port=29701)
    res = launch(worker, cfg, collect=True)
    for r in sorted(res):
        print(f"rank {r}: {res[r]}")
    ok = len(res) == 4 and all(v["roundtrip_ok"] and v["equiv_ok"] for v in res.values())
    print("=== SP-MoE token sharding: chunk/all-gather round-trip + per-token equivalence ===")
    print("ALL PASS" if ok else "FAIL")
