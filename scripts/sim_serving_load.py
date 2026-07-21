"""Simulate large-scale serving load: many concurrent requests through the engine
(continuous batching + paged incremental decode). Reports throughput/latency."""
import time, sys
from transformers import AutoTokenizer
from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

path="/data0/models/Qwen2.5-0.5B-Instruct"
N=int(sys.argv[1]) if len(sys.argv)>1 else 64
MAXTOK=32
tok=AutoTokenizer.from_pretrained(path)
templates=["The capital of France is","Explain photosynthesis:","2+2 equals",
           "Once upon a time","The largest planet is","Write a haiku about",
           "The speed of light is","In the year 2050,"]
prompts=[templates[i%len(templates)]+" "*(i%3) for i in range(N)]
pids=[tok(p).input_ids for p in prompts]
cfg=EngineConfig(model=ModelConfig(model_path=path),
                 cache=CacheConfig(block_size=16,num_blocks=8192),
                 scheduler=SchedulerConfig(max_num_seqs=256,max_num_batched_tokens=8192))
llm=LLM(cfg, executor=PagedNpuExecutor(path,8192,16))
t0=time.perf_counter()
outs=llm.generate(pids, max_tokens=MAXTOK)
dt=time.perf_counter()-t0
ntok=sum(len(o) for o in outs)
done=sum(1 for o in outs if len(o)>0)
print("=== SERVING LOAD SIM ===")
print(f"requests={N} completed={done} max_tokens={MAXTOK}")
print(f"wall={dt:.2f}s  throughput={N/dt:.1f} req/s, {ntok/dt:.0f} tok/s")
print(f"avg_out_tokens={ntok/N:.1f}  sample[0]={tok.decode(outs[0])!r}")
