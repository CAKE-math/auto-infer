"""Engine-level ACL-graph decode: GraphPagedNpuExecutor vs the HF-parity plain
executor through the full EngineCore (scheduler + paged KV + continuous batching).
Verifies decode-only steps run on a captured graph and produce identical tokens,
then prints graph/eager step counts + decode throughput.

  python scripts/smoke_graph_engine.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys
import time

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 4096


def cfg(path):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    base = ["The capital of France is", "2 + 2 =", "Once upon a time",
            "The meaning of life is", "Water is made of hydrogen and",
            "In the beginning", "Python is a programming", "The sun rises in the"]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2      # batch = 8*reps
    prompts = base * reps
    pids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    # graph-decode run
    gexec = GraphPagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE)
    t0 = time.perf_counter()
    graph_llm = LLM(cfg(path), executor=gexec)
    g_out = graph_llm.generate(pids, max_tokens=24)
    gt = time.perf_counter() - t0
    # same kernel, eager decode (isolates capture/replay correctness)
    eexec = GraphPagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE, force_eager=True)
    t0 = time.perf_counter()
    eager_llm = LLM(cfg(path), executor=eexec)
    e_out = eager_llm.generate(pids, max_tokens=24)
    et = time.perf_counter() - t0

    print("=== GRAPH-DECODE vs EAGER-FIA-v2 (same kernel) ===")
    ok = True
    for p, g, e in zip(prompts, g_out, e_out):
        match = g == e; ok = ok and match
        print(f"[{p!r}] graph==eager={match}")
        print(f"  out: {tok.decode(tok(p).input_ids + g)!r}")
    gs, es = gexec.runner.stats, eexec.runner.stats
    ntok = sum(len(o) for o in g_out)
    print(f"graph: steps(graph={gs['graph_steps']} eager={gs['eager_steps']}) {ntok/gt:.0f} tok/s")
    print(f"eager: steps(graph={es['graph_steps']} eager={es['eager_steps']}) {ntok/et:.0f} tok/s")
    print(f"decode speedup ~ {et/gt:.2f}x (incl. shared prefill)")
    print("ALL MATCH" if ok else "MISMATCH")
    graph_llm.close()
    eager_llm.close()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
