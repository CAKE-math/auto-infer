"""§6 two-level comm groups on REAL NPU (not just host planner): launch 2 ranks as
2 simulated nodes (AI_NNODES=2), have init_distributed BUILD the intra-node (HCCS)
and inter-node (RDMA) HCCL groups, and run a real collective over the INTER-NODE
group — proving the multi-node group construction is functional HCCL on Ascend.
The only residual for true multi-node is physical >=2 nodes + RoCE NICs (the
inter-node group here is intra-host since there is one physical node)."""
import os

from auto_infer.deploy.launcher import LauncherConfig, launch


def worker(rank, world_size, role):
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    os.environ["AI_NNODES"] = "2"          # 2 ranks => 2 simulated nodes (1 proc each)
    os.environ["AI_TP"] = "1"
    from auto_infer.distributed import parallel_state as ps
    ps.init_distributed()
    inter = ps.inter_node_group()
    assert inter is not None, "inter-node group not built"
    local = int(os.environ["LOCAL_RANK"])
    x = torch.ones(8, device=f"npu:{local}") * (rank + 1)
    dist.all_reduce(x, group=inter)        # collective over the INTER-NODE (RDMA) group
    dist.destroy_process_group()
    exp = sum(r + 1 for r in range(world_size))
    return {"rank": rank, "inter_allreduce": float(x[0].item()), "ok": abs(x[0].item() - exp) < 1e-3}


if __name__ == "__main__":
    cfg = LauncherConfig(nnodes=2, nproc_per_node=1, master_port=29631, role="engine")
    # NB: launching 2 ranks on this 1 physical node; AI_NNODES=2 makes them 2 logical nodes
    res = launch(worker, LauncherConfig(nnodes=1, nproc_per_node=2, master_port=29631), collect=True)
    for r in sorted(res):
        print(f"rank {r}: {res[r]}")
    ok = len(res) == 2 and all(v["ok"] for v in res.values())
    print("=== two-level comm groups functional on NPU (inter-node all_reduce) ===")
    print("ALL PASS" if ok else "FAIL")
