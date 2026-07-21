"""Baseline: vLLM-ascend decode throughput on the SAME model/box, to compare
against auto-infer. Runs vLLM's offline engine (graph mode = its default), greedy,
same prompt + max_tokens as our benches. NOT an auto_infer script — pure vLLM.

  python scripts/bench_vllm_ascend.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys
import time

from vllm import LLM, SamplingParams

N = 128
BATCHES = [1, 16]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    # low mem-util so it fits a small free window on the shared box
    llm = LLM(model=path, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=0.30, max_model_len=2048, enforce_eager=False)
    prompt = "Explain how a transformer decodes text."
    sp = SamplingParams(max_tokens=N, temperature=0.0)
    print(f"=== vLLM-ascend  model={path.split('/')[-1]}  max_tokens={N} ===")
    print(f"{'batch':>6} {'tok/s':>10}")
    for b in BATCHES:
        prompts = [prompt] * b
        llm.generate(prompts, sp)                      # warmup / graph capture
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            best = min(best, time.perf_counter() - t0)
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{b:>6} {gen / best:>10.1f}")


if __name__ == "__main__":
    main()
