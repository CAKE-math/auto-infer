"""Persistent-service stress: cancellation followed by repeated concurrent waves."""

import asyncio

from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                               ModelConfig, SchedulerConfig)
from auto_infer.engine.factory import build_executor
from auto_infer.serving.async_engine import AsyncEngine


PATH = "/data0/models/Qwen2.5-0.5B-Instruct"


async def main():
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    config = EngineConfig(
        model=ModelConfig(PATH),
        cache=CacheConfig(block_size=16, num_blocks=4096),
        scheduler=SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=4096),
        execution=ExecutionConfig(mode="paged", device_index=0))
    engine = AsyncEngine(config, build_executor(config))
    prompt = tokenizer("The capital of France is").input_ids
    cancelled = asyncio.create_task(engine.generate(prompt, max_tokens=128))
    await asyncio.sleep(0.01)
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass

    completed = 0
    for _ in range(5):
        results = await asyncio.gather(*[
            engine.generate(tokenizer(text).input_ids, max_tokens=8)
            for text in ("The capital of France is", "2 + 2 =", "Once upon a time") * 4
        ])
        assert all(len(result) == 8 for result in results)
        completed += len(results)
    assert engine.service.thread.is_alive()
    engine.close()
    assert not engine.service.thread.is_alive()
    print(f"SERVICE STABILITY PASS completed={completed} cancellation=1", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
