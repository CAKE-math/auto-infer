"""SP6: DeepSeek-V2-Lite ACL-graph decode (GraphMlaBackend, MLA+MoE captured)
vs eager FIA-v2 (same kernel, force_eager=True) through the full engine —
mirrors `scripts/smoke_graph_engine.py`'s Qwen2 graph==eager engine-level
parity check, and is the batched multi-prompt analogue of
`scripts/verify_qwen2_graphdecode_batched.py` for DeepSeek: several prompts
decoded together inside captured gears, each compared token-for-token against
the eager reference. Uses `GraphPagedNpuExecutor` unchanged (SP6 made it
model-agnostic via `model.make_graph_backend`) — no hand-rolled MLA/MoE ops
here, so this exercises the REAL `GraphMlaBackend` + `DeepseekV2Model.forward`
(dense+MoE FFN) path end-to-end, same as production would.

  python scripts/verify_deepseek_graphdecode.py /data1/models/DeepSeek-V2-Lite-Chat
"""
import os

# See run_deepseek_chat.py: expandable_segments avoids the NPU allocator
# fragmentation that otherwise OOMs DeepSeek's MoE. Set before torch_npu init.
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import sys
import time

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 2048


def cfg(path):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    base = ["The capital of France is", "2 + 2 =", "Once upon a time",
            "Water is made of hydrogen and"]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1      # batch = 4*reps
    prompts = base * reps
    pids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    # Two 16B graph-capable executors can't co-reside on one card (a captured
    # NPUGraph pins its memory pool — del+empty_cache won't free it), so run them
    # on SEPARATE cards: graph on device 0, eager on device 1. Set
    # ASCEND_RT_VISIBLE_DEVICES to two free cards (e.g. 0,1).
    gexec = GraphPagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE, max_gear=8, device_index=0)
    t0 = time.perf_counter()
    g_out = LLM(cfg(path), executor=gexec).generate(pids, max_tokens=24)
    gt = time.perf_counter() - t0
    gs = dict(gexec.runner.stats)
    # same kernel, eager decode (isolates capture/replay correctness — force_eager
    # still routes through GraphMlaBackend's plain .out call, capturing=False)
    eexec = GraphPagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE, max_gear=8,
                                  force_eager=True, device_index=1)
    t0 = time.perf_counter()
    e_out = LLM(cfg(path), executor=eexec).generate(pids, max_tokens=24)
    et = time.perf_counter() - t0
    es = dict(eexec.runner.stats)

    print("=== DeepSeek-V2-Lite GRAPH-DECODE (MLA+MoE) vs EAGER-FIA-v2 (same kernel) ===")
    ok = True
    for p, g, e in zip(prompts, g_out, e_out):
        match = g == e; ok = ok and match
        print(f"[{p!r}] graph==eager={match}")
        print(f"  out: {tok.decode(tok(p).input_ids + g, skip_special_tokens=True)!r}")
    ntok = sum(len(o) for o in g_out)
    print(f"graph: steps(graph={gs['graph_steps']} eager={gs['eager_steps']}) {ntok/gt:.0f} tok/s")
    print(f"eager: steps(graph={es['graph_steps']} eager={es['eager_steps']}) {ntok/et:.0f} tok/s")
    print(f"decode speedup ~ {et/gt:.2f}x (incl. shared prefill)")
    print("ALL MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
