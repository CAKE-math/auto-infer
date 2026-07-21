"""V3-architecture (Moonlight-16B-A3B, DeepseekV3ForCausalLM, sigmoid/noaux_tc) through
the FULL paged engine on real MLA dims: PagedNpuExecutor + EngineCore (scheduler +
paged MLA KV + continuous batching + async scheduling). Trained weights => checks
COHERENT greedy output through the engine, the engine-level counterpart to the bare
forward sign-off (verify_v3_gating_trained.py). Real dims clear the FIA tiny-dim limit
that gated deepseek-v3-tiny-random."""
from transformers import AutoTokenizer
import sys

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

PATH = "/data2/models/Moonlight-16B-A3B-Instruct"
DEVICE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BLOCK, NB = 16, 512
tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
ids = tok("The capital of France is", add_special_tokens=False).input_ids
if tok.bos_token_id is not None:
    ids = [tok.bos_token_id] + ids

cfg = EngineConfig(model=ModelConfig(model_path=PATH),
                   cache=CacheConfig(block_size=BLOCK, num_blocks=NB),
                   scheduler=SchedulerConfig(max_num_batched_tokens=2048),
                   async_scheduling=True)
llm = LLM(cfg, executor=PagedNpuExecutor(
    PATH, NB, BLOCK, device_index=DEVICE))
out = llm.generate([ids], max_tokens=12)
llm.close()
gen = tok.decode(out[0])
print(f"paged-engine gen: {gen!r}")
coherent = any(c.isalpha() for c in gen) and len(set(gen.split())) > 2
print("=== V3 (Moonlight) through full paged EngineCore ===")
print("COHERENT" if coherent else "GARBAGE")
if not coherent:
    raise SystemExit(1)
