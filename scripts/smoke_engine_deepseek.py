"""Full EngineCore loop driving the model-agnostic NpuExecutor on DeepSeek-V2-Lite.
  python scripts/smoke_engine_deepseek.py /data1/models/DeepSeek-V2-Lite-Chat
"""
import sys
from transformers import AutoTokenizer
from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.npu_executor import NpuExecutor
from auto_infer.entrypoints.llm import LLM


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    prompts = ["The capital of France is", "The largest planet in the solar system is"]
    pids = [[tok.bos_token_id] + tok(p).input_ids for p in prompts]  # DeepSeek needs explicit BOS
    cfg = EngineConfig(model=ModelConfig(model_path=path),
                       cache=CacheConfig(block_size=16, num_blocks=4096),
                       scheduler=SchedulerConfig(max_num_batched_tokens=4096))
    llm = LLM(cfg, executor=NpuExecutor(path))
    outs = llm.generate(pids, max_tokens=8)
    print("=== ENGINE + DeepSeek-V2 MLA+MoE END-TO-END ===")
    for p, o in zip(prompts, outs):
        print(f"[{p!r}] -> {tok.decode(o)!r}")


if __name__ == "__main__":
    main()
