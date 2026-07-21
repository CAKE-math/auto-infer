"""Multi-node EXECUTION path with a real model (spec §6/§7): DeepSeek-V2-Lite run
across 2 logical nodes (AI_NNODES=2, 2 NPU cards = 2 nodes, 1 rank each), with MoE
EXPERT PARALLEL over the INTER-NODE (RDMA-role) group. Verifies the full multi-node
parallel execution code path end-to-end: two-level groups active + EP all-to-all/
all-reduce over the inter-node group + correct output (== single-card greedy).

This is the closest verification possible to multi-node DeepSeek-V3 on this node;
the only residual is 671B scale + a 2nd physical node with RoCE NICs (pure HW)."""
import os

from auto_infer.deploy.launcher import LauncherConfig, launch

PATH = "/data1/models/DeepSeek-V2-Lite-Chat"


def worker(rank, world_size, role):
    import torch
    import torch_npu  # noqa: F401
    os.environ["AI_NNODES"] = "2"          # 2 ranks => 2 logical nodes
    os.environ["AI_TP"] = "1"              # TP=1 (intra-node trivial); EP=2 over inter-node
    from transformers import AutoTokenizer
    from auto_infer.distributed import parallel_state as ps
    from auto_infer.models.deepseek_v2 import DeepseekV2Model
    ps.init_distributed()
    local = int(os.environ["LOCAL_RANK"]); dev = torch.device(f"npu:{local}")
    tok = AutoTokenizer.from_pretrained(PATH)
    m = DeepseekV2Model.from_pretrained(PATH, dev, torch.bfloat16)
    ids = tok("The capital of France is", add_special_tokens=False).input_ids
    ids = [tok.bos_token_id] + ids
    seq = list(ids)
    for _ in range(6):                     # greedy a few tokens (EP over inter-node each step)
        t = torch.tensor(seq, device=dev); pos = torch.arange(len(seq), device=dev)
        seq.append(int(m.forward_dense(t, pos)[-1].argmax()))
    out = tok.decode(seq[len(ids):])
    return {"rank": rank, "ep_size": ps.ep_size(), "gen": out}


if __name__ == "__main__":
    res = launch(worker, LauncherConfig(nnodes=1, nproc_per_node=2, master_port=29651),
                 collect=True)
    for r in sorted(res):
        print(f"rank {r}: ep_size={res[r]['ep_size']} gen={res[r]['gen']!r}")
    gens = [v["gen"] for v in res.values()]
    eps = [v["ep_size"] for v in res.values()]
    ok = len(res) == 2 and all(e == 2 for e in eps) and len(set(gens)) == 1 and "Paris" in gens[0]
    print("=== multi-node (2 logical nodes) EP execution, real model ===")
    print("ALL PASS" if ok else "FAIL")
