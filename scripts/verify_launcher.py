"""deploy/launcher verification: minimal rendezvous launcher spawns N worker
processes with the env contract (RANK/WORLD_SIZE/...), and the workers form a real
torch.distributed group over the SAME rendezvous (gloo/localhost — no NICs/NPU
needed) and all_reduce, proving the launch-agnostic env contract end-to-end.
"""
import os

from auto_infer.deploy.launcher import LauncherConfig, launch


def worker(rank, world_size, role):
    import torch
    import torch.distributed as dist
    # workers read the SAME env the launcher exported (also reachable via torchrun/K8s)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    t = torch.tensor([rank + 1.0])
    dist.all_reduce(t)
    dist.destroy_process_group()
    expected = world_size * (world_size + 1) / 2          # sum of 1..world_size
    return {"role": role, "local_rank": os.environ["LOCAL_RANK"],
            "allreduce": float(t.item()), "ok": abs(t.item() - expected) < 1e-5}


if __name__ == "__main__":
    cfg = LauncherConfig(nnodes=1, nproc_per_node=4, master_port=29571, role="engine")
    res = launch(worker, cfg, collect=True)
    for rank in sorted(res):
        print(f"rank {rank}: {res[rank]}")
    ok = len(res) == 4 and all(r["ok"] for r in res.values())
    print("=== launcher rendezvous + all_reduce ===")
    print("ALL PASS" if ok else "FAIL")
