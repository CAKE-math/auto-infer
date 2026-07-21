import os, torch, torch_npu
import torch.distributed as dist
local = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"])
torch.npu.set_device(local)
dist.init_process_group(backend="hccl")
ws = dist.get_world_size()
x = torch.ones(4, device=f"npu:{local}") * (rank + 1)
dist.all_reduce(x)
exp = sum(r + 1 for r in range(ws))
if rank == 0:
    print(f"HCCL all_reduce result={x[0].item()} expect={exp} ->", "OK" if abs(x[0].item()-exp)<1e-3 else "FAIL")
dist.destroy_process_group()
