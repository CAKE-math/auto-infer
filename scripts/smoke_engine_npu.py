"""End-to-end: full EngineCore loop driving the real NPU executor.

  python scripts/smoke_engine_npu.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.npu_executor import NpuExecutor
from auto_infer.entrypoints.llm import LLM


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    prompts = ["The capital of France is", "2 + 2 ="]
    prompt_ids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    cfg = EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=16, num_blocks=4096),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096),
    )
    llm = LLM(cfg, executor=NpuExecutor(path))
    outs = llm.generate(prompt_ids, max_tokens=16)
    print("=== ENGINE + NPU END-TO-END ===")
    for p, o in zip(prompts, outs):
        full = tok.decode(tok(p).input_ids + o)
        print(f"[{p!r}] -> {full!r}")


if __name__ == "__main__":
    main()
