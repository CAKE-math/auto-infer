"""Paged FIA multi-batch correctness: compare paged executor vs the
HF-parity-verified plain executor through the full engine.

  python scripts/smoke_paged.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.npu_executor import NpuExecutor
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 4096


def make_cfg(path):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096),
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time"]
    pids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    paged = LLM(make_cfg(path), executor=PagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE))
    plain = LLM(make_cfg(path), executor=NpuExecutor(path))
    try:
        paged_out = paged.generate(pids, max_tokens=16)
        plain_out = plain.generate(pids, max_tokens=16)
    finally:
        paged.close()
        plain.close()

    print("=== PAGED FIA (multi-batch) vs PLAIN ===")
    ok = True
    for p, pg, pl in zip(prompts, paged_out, plain_out):
        match = pg == pl
        ok = ok and match
        print(f"[{p!r}]")
        print(f"  paged: {tok.decode(tok(p).input_ids + pg)!r}")
        print(f"  plain: {tok.decode(tok(p).input_ids + pl)!r}")
        print(f"  match-plain: {match}")
    print("ALL MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
