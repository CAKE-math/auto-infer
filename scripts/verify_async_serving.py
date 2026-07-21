"""Async serving: N concurrent asyncio clients served by one engine loop (continuous batching)."""
import asyncio, time
import torch_npu  # noqa
from transformers import AutoTokenizer
from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.serving.async_engine import AsyncEngine
from auto_infer.worker.model_runner import PagedNpuExecutor

async def main():
    path="/data0/models/Qwen2.5-0.5B-Instruct"
    tok=AutoTokenizer.from_pretrained(path)
    cfg=EngineConfig(model=ModelConfig(model_path=path),
                     cache=CacheConfig(block_size=16,num_blocks=8192),
                     scheduler=SchedulerConfig(max_num_seqs=256,max_num_batched_tokens=8192))
    eng=AsyncEngine(cfg, PagedNpuExecutor(path,8192,16))
    templates=["The capital of France is","Once upon a time","2+2 equals","The sun is",
               "Explain gravity:","In 2050,","The ocean is","Light travels"]
    N=48
    pids=[tok(templates[i%len(templates)]).input_ids for i in range(N)]
    t0=time.perf_counter()
    results=await asyncio.gather(*[eng.generate(ids,24) for ids in pids])   # N concurrent async clients
    dt=time.perf_counter()-t0
    ntok=sum(len(r) for r in results); done=sum(1 for r in results if r)
    print(f"=== ASYNC SERVING: {N} concurrent clients, 1 engine loop ===")
    print(f"completed={done}/{N} wall={dt:.2f}s tok/s={ntok/dt:.0f}")
    print("sample:", repr(tok.decode(results[0])))
    eng.close()
    if done != N:
        raise SystemExit(1)

asyncio.run(main())
