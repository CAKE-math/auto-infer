"""Launch-agnostic deploy seam (spec sec 12/14). MVP = minimal multi-node
launcher: env rendezvous (MASTER_ADDR/PORT, RANK, WORLD_SIZE, LOCAL_RANK) + per-
node role spawn (prefill / decode / proxy) from one cluster spec. Ansible / K8s+
Volcano+LWS are OPTIONAL external adapters; the framework depends only on this env
contract, so any orchestrator that sets these vars + launches `python -m ... ` per
rank works unchanged.

The framework code (worker.py / parallel_state) reads the SAME env vars whether
launched here, by torchrun, or by K8s — that is the launch-agnostic property.
"""
import multiprocessing as mp
import os
from dataclasses import dataclass


@dataclass
class LauncherConfig:
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    role: str = "engine"          # engine | prefill | decode | proxy


def _set_rendezvous_env(cfg: LauncherConfig, local_rank: int) -> dict:
    """Compute this process's global rank/world_size and export the env contract."""
    world_size = cfg.nnodes * cfg.nproc_per_node
    rank = cfg.node_rank * cfg.nproc_per_node + local_rank
    env = {
        "MASTER_ADDR": cfg.master_addr,
        "MASTER_PORT": str(cfg.master_port),
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "LOCAL_RANK": str(local_rank),
        "AI_ROLE": cfg.role,
    }
    os.environ.update(env)
    return env


def _entry(local_rank, cfg, worker_fn, q):
    env = _set_rendezvous_env(cfg, local_rank)
    result = worker_fn(int(env["RANK"]), int(env["WORLD_SIZE"]), cfg.role)
    if q is not None:
        q.put((int(env["RANK"]), result))


def launch(worker_fn, cfg: LauncherConfig, collect: bool = False):
    """Spawn cfg.nproc_per_node worker processes on THIS node, each with the
    rendezvous env set, calling worker_fn(rank, world_size, role). Multi-node = run
    this on each node with the matching node_rank (same master_addr/port)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue() if collect else None
    procs = []
    for lr in range(cfg.nproc_per_node):
        p = ctx.Process(target=_entry, args=(lr, cfg, worker_fn, q), daemon=False)
        p.start()
        procs.append(p)
    results = {}
    if collect:
        for _ in procs:
            rank, res = q.get()
            results[rank] = res
    for p in procs:
        p.join()
    return results
