"""STRICT 对拍: SP×EP mesh output == single-card full-MoE output (spec §6/§10
token equivalence). Two launches on real NPU:
  ref : 1 card, no sharding (sp=1, ep=1) -> naive full MoE (the ground truth)
  mesh: 4 cards, AI_MESH_SP=2 × AI_MESH_EP=2 -> SP token-shard + EP expert-shard
Compares, for the same prompt: (a) last-token logit difference as a diagnostic,
(b) argmax token identical, (c) 12-token greedy sequence identical. A correct
SP×EP composition must reproduce the single-card result, not merely stay coherent.
"""
import os

from auto_infer.deploy.launcher import LauncherConfig, launch

PATH = "/data1/models/DeepSeek-V2-Lite-Chat"
NGEN = 12


def worker(rank, world_size, role):
    import torch
    import torch_npu  # noqa: F401
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    if role == "mesh":
        os.environ["AI_TP"] = "1"
        os.environ["AI_MESH_SP"] = "2"
        os.environ["AI_MESH_EP"] = "2"
    from transformers import AutoTokenizer
    from auto_infer.distributed import parallel_state as ps
    from auto_infer.models.deepseek_v2 import DeepseekV2Model
    ps.init_distributed()
    dev = torch.device(f"npu:{local}")
    tok = AutoTokenizer.from_pretrained(PATH)
    m = DeepseekV2Model.from_pretrained(PATH, dev, torch.bfloat16,
                                        ep_size=ps.ep_size(), ep_rank=ps.ep_rank())
    ids = [tok.bos_token_id] + tok("The capital of France is", add_special_tokens=False).input_ids
    # single fixed-input forward -> last-token logits (fp32, for a clean compare)
    t = torch.tensor(ids, device=dev); pos = torch.arange(len(ids), device=dev)
    last_logits = m.forward_dense(t, pos)[-1].float().cpu()
    # greedy sequence
    seq = list(ids)
    for _ in range(NGEN):
        tt = torch.tensor(seq, device=dev); pp = torch.arange(len(seq), device=dev)
        seq.append(int(m.forward_dense(tt, pp)[-1].argmax()))
    return {"rank": rank, "role": role, "sp": ps.sp_size(), "ep": ps.ep_size(),
            "argmax": int(last_logits.argmax()), "logits": last_logits.tolist(), "seq": seq[len(ids):]}


if __name__ == "__main__":
    import torch
    ref = launch(worker, LauncherConfig(nproc_per_node=1, master_port=29741, role="ref"),
                 collect=True)[0]
    mesh = launch(worker, LauncherConfig(nproc_per_node=4, master_port=29743, role="mesh"),
                  collect=True)
    print(f"ref : sp={ref['sp']} ep={ref['ep']} argmax={ref['argmax']} seq={ref['seq']}")
    ok = True
    for r in sorted(mesh):
        v = mesh[r]
        vl = torch.tensor(v["logits"]); rl = torch.tensor(ref["logits"])
        max_abs = float((vl - rl).abs().max())
        allclose = torch.allclose(vl, rl, atol=5e-2, rtol=1e-2)
        argmax_eq = v["argmax"] == ref["argmax"]
        seq_eq = v["seq"] == ref["seq"]
        ok = ok and argmax_eq and seq_eq
        print(f"mesh r{r}: sp={v['sp']} ep={v['ep']} argmax_eq={argmax_eq} "
              f"seq_eq={seq_eq} max|Δlogit|={max_abs:.4f} allclose={allclose}")
    print("=== STRICT 对拍: SP×EP mesh == single-card full-MoE ===")
    print("ALL MATCH" if ok else "MISMATCH")
