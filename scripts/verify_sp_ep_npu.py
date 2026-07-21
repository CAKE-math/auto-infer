"""SP×EP device mesh on REAL NPU (spec §6): DeepSeek-V2-Lite on 4 cards laid out as
a 2×2 mesh (AI_MESH_SP=2, AI_MESH_EP=2) — SP shards MoE tokens on one axis, EP
shards experts on the orthogonal axis, composing like omni's use_sequence_parallel_moe
+ EP. Each rank loads only its EP-expert shard (sharded loader §15.1). Correct
composition => coherent greedy output ("Paris"); a collision bug => garbage.
Launched via our rendezvous launcher.
"""
import os

from auto_infer.deploy.launcher import LauncherConfig, launch

PATH = "/data1/models/DeepSeek-V2-Lite-Chat"


def worker(rank, world_size, role):
    import torch
    import torch_npu  # noqa: F401
    os.environ["AI_TP"] = "1"                      # attention replicated (DP); mesh drives MoE
    os.environ["AI_MESH_SP"] = "2"
    os.environ["AI_MESH_EP"] = "2"
    from transformers import AutoTokenizer
    from auto_infer.distributed import parallel_state as ps
    from auto_infer.models.deepseek_v2 import DeepseekV2Model
    ps.init_distributed()
    local = int(os.environ["LOCAL_RANK"]); dev = torch.device(f"npu:{local}")
    tok = AutoTokenizer.from_pretrained(PATH)
    # each rank loads ONLY its EP-expert shard (loader §15.1) using the mesh EP coord
    m = DeepseekV2Model.from_pretrained(PATH, dev, torch.bfloat16,
                                        ep_size=ps.ep_size(), ep_rank=ps.ep_rank())
    ids = [tok.bos_token_id] + tok("The capital of France is", add_special_tokens=False).input_ids
    seq = list(ids)
    for _ in range(6):
        t = torch.tensor(seq, device=dev); pos = torch.arange(len(seq), device=dev)
        seq.append(int(m.forward_dense(t, pos)[-1].argmax()))
    return {"rank": rank, "sp": ps.sp_size(), "sp_rank": ps.sp_rank(),
            "ep": ps.ep_size(), "ep_rank": ps.ep_rank(),
            "gen": tok.decode(seq[len(ids):])}


if __name__ == "__main__":
    res = launch(worker, LauncherConfig(nnodes=1, nproc_per_node=4, master_port=29721),
                 collect=True)
    for r in sorted(res):
        v = res[r]
        print(f"rank {r}: sp={v['sp']}(r{v['sp_rank']}) ep={v['ep']}(r{v['ep_rank']}) gen={v['gen']!r}")
    gens = [v["gen"] for v in res.values()]
    ok = len(res) == 4 and all("Paris" in g for g in gens) and len(set(gens)) == 1
    print("=== SP×EP mesh (2×2) DeepSeek-V2-Lite on 4 NPU ===")
    print("ALL PASS" if ok else "FAIL")
