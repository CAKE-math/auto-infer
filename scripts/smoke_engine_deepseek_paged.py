"""DeepSeek-V2 MLA+MoE, PAGED FIA, MULTI-BATCH, through full EngineCore."""
import sys
from transformers import AutoTokenizer
from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

BS, NB = 16, 4096
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    prompts = ["The capital of France is", "1 + 1 =", "The sun rises in the"]
    pids = [[tok.bos_token_id] + tok(p).input_ids for p in prompts]
    cfg = EngineConfig(model=ModelConfig(model_path=path),
                       cache=CacheConfig(block_size=BS, num_blocks=NB),
                       scheduler=SchedulerConfig(max_num_batched_tokens=4096))
    llm = LLM(cfg, executor=PagedNpuExecutor(path, NB, BS))
    outs = llm.generate(pids, max_tokens=8)
    print("=== DeepSeek-V2 MLA+MoE PAGED multi-batch via EngineCore ===")
    for p, o in zip(prompts, outs):
        print(f"[{p!r}] -> {tok.decode(o)!r}")
if __name__ == "__main__":
    main()
