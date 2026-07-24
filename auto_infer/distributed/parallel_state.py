"""Runtime parallel groups derived from one :class:`ParallelMesh`."""
import os
import torch

_TP_SIZE = 1
_TP_RANK = 0
_DP_SIZE = 1
_SP_SIZE = 1             # sequence-parallel size; SP shares the TP group (Megatron)
_TP_GROUP = None
_INITED = False

# Optional SP×EP device mesh (spec §6 "attention-DP + MoE-EP 混合切"). When set,
# SP (token sharding) and EP (expert sharding) live on DISTINCT, orthogonal axes so
# they compose — matching omni-npu's use_sequence_parallel_moe + EP. Left None for
# the default single-axis EP-via-TP path (which stays exactly as verified).
_SP_GROUP = None
_SP_RANK = 0
_EP_GROUP = None
_EP_RANK = 0
_EP_SIZE = 1
_EP_HCCL_COMM_NAME = None
_DP_GROUP = None
_CP_GROUP = None
_CP_RANK = 0
_CP_SIZE = 1


def plan_two_level_groups(nnodes: int, nproc_per_node: int) -> dict:
    """Host-side topology utility for deployment planning.

    intra[node] = ranks co-located on one node (TP / dense all-reduce over HCCS).
    inter[lr]   = ranks sharing a local position across nodes (EP all-to-all over RDMA).
    """
    ws = nnodes * nproc_per_node
    intra = [list(range(n * nproc_per_node, (n + 1) * nproc_per_node)) for n in range(nnodes)]
    inter = [list(range(lr, ws, nproc_per_node)) for lr in range(nproc_per_node)]
    return {"intra": intra, "inter": inter, "world_size": ws}


def init_distributed(pconfig):
    """Create every HCCL group from the configured mesh exactly once."""
    global _TP_SIZE, _TP_RANK, _DP_SIZE, _SP_SIZE, _TP_GROUP, _INITED
    global _DP_GROUP, _CP_GROUP, _CP_RANK, _CP_SIZE
    global _SP_GROUP, _SP_RANK, _EP_GROUP, _EP_RANK, _EP_SIZE
    global _EP_HCCL_COMM_NAME
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    mesh = pconfig.to_mesh()
    mesh.validate(ws)
    if ws == 1:
        return
    if _INITED:      # idempotent across rebuilt EngineCore instances
        current = (_TP_SIZE, _DP_SIZE, _EP_SIZE, _CP_SIZE, _SP_SIZE)
        requested = (mesh.tp, mesh.dp, mesh.ep, mesh.cp, mesh.sp)
        if current != requested:
            from auto_infer.errors import ConfigurationError
            raise ConfigurationError(
                f"distributed runtime already initialized as {current}; "
                f"requested {requested}")
        return       # must not re-create HCCL groups (leaks the old ones)
    from importlib import import_module
    import_module("torch_npu")  # registers torch.npu and the HCCL backend
    import torch.distributed as dist
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    selected = {}
    for axis in ("tp", "dp", "ep", "cp", "sp"):
        axis_size = getattr(mesh, axis)
        if axis_size == 1:
            selected[axis] = (None, 0, 1)
            continue
        for members in mesh.groups(axis):
            group = dist.new_group(members)
            if rank in members:
                selected[axis] = (group, members.index(rank), axis_size)
    _TP_GROUP, _TP_RANK, _TP_SIZE = selected["tp"]
    _DP_GROUP, _, _DP_SIZE = selected["dp"]
    _EP_GROUP, _EP_RANK, _EP_SIZE = selected["ep"]
    if _EP_SIZE > 1:
        backend = _EP_GROUP._get_backend(torch.device("npu"))
        _EP_HCCL_COMM_NAME = backend.get_hccl_comm_name(_EP_RANK)
    _CP_GROUP, _CP_RANK, _CP_SIZE = selected["cp"]
    _SP_GROUP, _SP_RANK, _SP_SIZE = selected["sp"]
    _INITED = True


def tp_size() -> int:
    return _TP_SIZE


def tp_rank() -> int:
    return _TP_RANK


def dp_size() -> int:
    return _DP_SIZE


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    if _TP_SIZE == 1:
        return x
    import torch.distributed as dist
    dist.all_reduce(x, group=_TP_GROUP)
    return x


def tp_barrier() -> None:
    """Synchronize the tensor-parallel ranks; a free no-op at TP1."""
    if _TP_SIZE == 1:
        return
    import torch.distributed as dist
    dist.barrier(group=_TP_GROUP)


def _ep_group():
    return _EP_GROUP


def ep_size() -> int:
    return _EP_SIZE


def ep_rank() -> int:
    return _EP_RANK


def ep_topology():
    """Return the cached resources required by fused EP communication."""
    from auto_infer.distributed.topology import ExpertParallelTopology
    return ExpertParallelTopology(
        group=_EP_GROUP,
        rank=_EP_RANK,
        world_size=_EP_SIZE,
        hccl_comm_name=_EP_HCCL_COMM_NAME,
    )


def _sp_group():
    return _SP_GROUP if _SP_GROUP is not None else _TP_GROUP


def sp_size() -> int:
    """Sequence-parallel size. SP-MoE (spec §6, matches omni-npu/omni-models
    `use_sequence_parallel_moe` used by DeepSeek-V3/Qwen3-MoE/GLM4-MoE): shard the
    MoE's tokens along the sequence dim. Uses the dedicated SP mesh axis when a
    mesh is configured, else the TP group (Megatron single-axis)."""
    return _SP_SIZE


def sp_rank() -> int:
    return _SP_RANK


def sp_chunk(x: torch.Tensor):
    """Enter the SP-MoE region: pad tokens (dim 0) up to a multiple of sp_size,
    then keep ONLY this rank's contiguous 1/sp token shard. Returns
    ``(local_shard, num_tokens)``; num_tokens is needed to unpad after all-gather.
    Matches vLLM's ``sequence_parallel_chunk`` semantics used by omni-npu."""
    n = _SP_SIZE
    num_tokens = x.shape[0]
    if n == 1:
        return x, num_tokens
    pad = (-num_tokens) % n
    if pad:
        x = torch.cat([x, x.new_zeros((pad, *x.shape[1:]))], dim=0)
    shard = x.shape[0] // n
    r = sp_rank()
    return x[r * shard:(r + 1) * shard], num_tokens


def sp_all_gather(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Leave the SP-MoE region: all-gather the per-rank token shards back into the
    full (padded) sequence across the SP group, then drop the padding."""
    n = _SP_SIZE
    if n == 1:
        return x
    import torch.distributed as dist
    parts = [torch.empty_like(x) for _ in range(n)]
    dist.all_gather(parts, x.contiguous(), group=_sp_group())
    return torch.cat(parts, dim=0)[:num_tokens]


def cp_all_gather(x: torch.Tensor) -> torch.Tensor:
    if _CP_SIZE == 1:
        return x
    import torch.distributed as dist
    parts = [torch.empty_like(x) for _ in range(_CP_SIZE)]
    dist.all_gather(parts, x.contiguous(), group=_CP_GROUP)
    return torch.cat(parts, dim=0)
