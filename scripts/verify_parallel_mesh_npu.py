"""Verify config-driven orthogonal SP=2, EP=2 HCCL groups on four NPUs."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from auto_infer.config import ParallelConfig
from auto_infer.distributed import parallel_state as ps


local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
ps.init_distributed(ParallelConfig(sp_size=2, ep_size=2))
rank = dist.get_rank()

ep_value = torch.tensor(float(rank), device=f"npu:{local_rank}")
dist.all_reduce(ep_value, group=ps.ep_topology().group)
expected_ep = 1.0 if rank < 2 else 5.0
assert ep_value.item() == expected_ep, (rank, ep_value.item(), expected_ep)

sp_value = torch.tensor([[float(rank)]], device=f"npu:{local_rank}")
sp_values = ps.sp_all_gather(sp_value, num_tokens=2).flatten().cpu().tolist()
expected_sp = [0.0, 2.0] if rank % 2 == 0 else [1.0, 3.0]
assert sp_values == expected_sp, (rank, sp_values, expected_sp)

print(f"rank={rank} ep_sum={ep_value.item():.0f} sp_values={sp_values}", flush=True)
dist.barrier()
if rank == 0:
    print("PARALLEL MESH PASS", flush=True)
dist.destroy_process_group()
