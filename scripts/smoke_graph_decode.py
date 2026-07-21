"""Generic ACL-graph decode smoke: drive any registered model through
GraphPagedNpuExecutor (graph) vs force_eager, token-for-token. Arch-dispatched
(GraphPagedNpuExecutor picks the model class from config.architectures), so it
works for Qwen2/Qwen3 dense and DeepSeek MLA+MoE alike.

  python scripts/smoke_graph_decode.py /data1/models/Qwen3-0.6B [reps]
"""
import sys

import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK, NB = 16, 2048


def cfg(path):
    return EngineConfig(model=ModelConfig(model_path=path),
                        cache=CacheConfig(block_size=BLOCK, num_blocks=NB),
                        scheduler=SchedulerConfig(max_num_batched_tokens=4096))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tok = AutoTokenizer.from_pretrained(path)
    base = ["The capital of France is", "2 + 2 =", "Once upon a time",
            "Water is made of hydrogen and"]
    prompts = base * reps
    pids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    gexec = GraphPagedNpuExecutor(path, NB, BLOCK, max_gear=8)
    g_out = LLM(cfg(path), executor=gexec).generate(pids, max_tokens=24)
    gs = dict(gexec.runner.stats)
    eexec = GraphPagedNpuExecutor(path, NB, BLOCK, max_gear=8, force_eager=True)
    e_out = LLM(cfg(path), executor=eexec).generate(pids, max_tokens=24)
    print(f"=== GRAPH DECODE vs EAGER ({path.split('/')[-1]}) ===")
    ok = True
    for p, g, e in zip(prompts, g_out, e_out):
        m = g == e; ok = ok and m
        print(f"  graph==eager={m}  {tok.decode(tok(p).input_ids + g)!r}")
    print(f"graph steps: graph={gs['graph_steps']} eager={gs['eager_steps']}")
    print("ALL MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
