"""Run DeepSeek-V2-Lite-Chat end-to-end through the engine (MLA + MoE via the
unified forward(ctx) + MlaFIABackend/GraphMlaBackend), with the tokenizer's
chat template.

  python scripts/run_deepseek_chat.py /data1/models/DeepSeek-V2-Lite-Chat          # eager
  python scripts/run_deepseek_chat.py /data1/models/DeepSeek-V2-Lite-Chat graph    # SP6 ACL-graph decode
"""
import os

# DeepSeek's 16B MoE + paged KV fragments the NPU allocator (reserved >> allocated),
# so a small fused-MoE alloc can OOM on a card that mem_get_info() reports as free.
# expandable_segments removes the fragmentation gap. Must be set before torch_npu's
# first allocation (i.e. here, before any import that touches the device); setdefault
# lets an explicit env var still override.
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import sys

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

BLOCK, NB = 16, 2048


def _graph_executor(path):
    from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor
    return GraphPagedNpuExecutor(path, NB, BLOCK, max_gear=8)


EXECUTORS = {
    "eager": lambda path: PagedNpuExecutor(path, NB, BLOCK),
    "graph": _graph_executor,
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    executor_name = sys.argv[2] if len(sys.argv) > 2 else "eager"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    questions = [
        "What is the capital of France, and what is it famous for?",
        "Write one sentence explaining what attention is in a transformer.",
    ]
    def to_ids(enc):
        if hasattr(enc, "input_ids"):        # BatchEncoding
            enc = enc.input_ids
        if isinstance(enc, dict):
            enc = enc["input_ids"]
        if hasattr(enc, "tolist"):           # tensor
            enc = enc.tolist()
        while enc and isinstance(enc[0], list):   # unbatch
            enc = enc[0]
        return [int(x) for x in enc]

    prompts = []
    for q in questions:
        try:
            enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                          add_generation_prompt=True, tokenize=True)
        except Exception:
            enc = tok(q).input_ids
        prompts.append(to_ids(enc))

    cfg = EngineConfig(model=ModelConfig(model_path=path),
                       cache=CacheConfig(block_size=BLOCK, num_blocks=NB),
                       scheduler=SchedulerConfig(max_num_batched_tokens=4096))
    llm = LLM(cfg, executor=EXECUTORS[executor_name](path))
    outs = llm.generate([list(p) for p in prompts], max_tokens=80)
    print(f"=== DeepSeek-V2-Lite-Chat (engine, MLA+MoE via forward(ctx), executor={executor_name!r}) ===")
    for q, o in zip(questions, outs):
        print(f"\nQ: {q}\nA: {tok.decode(o, skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
