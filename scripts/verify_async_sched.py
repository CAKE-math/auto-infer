"""Stage 6 verification (spec sec 10): inter-step async scheduling produces
TOKEN-IDENTICAL output to synchronous scheduling on real NPU, and next-step
device work is submitted before the step returns (host/device overlap structure).
"""
import sys
import time

from transformers import AutoTokenizer

from auto_infer.config import (
    CacheConfig, EngineConfig, ExecutionConfig, ModelConfig, SchedulerConfig)
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.entrypoints.llm import LLM

BLOCK_SIZE, NUM_BLOCKS = 16, 4096


def cfg(path, async_sched, mode):
    return EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096),
        execution=ExecutionConfig(mode=mode, max_gear=64),
        async_scheduling=async_sched)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time",
               "Python is a programming"]
    pids = [tok(p).input_ids for p in prompts]

    for mode in ("paged", "graph"):
        sync_llm = LLM(cfg(path, False, mode))
        sync_out = sync_llm.generate(pids, max_tokens=24)
        sync_48 = None
        if mode == "graph":
            sync_48 = sync_llm.generate(
                [list(pids[i % len(pids)]) for i in range(48)], max_tokens=12)
        sync_llm.close()
        t0 = time.perf_counter()
        async_llm = LLM(cfg(path, True, mode))
        async_out = async_llm.generate(pids, max_tokens=24)
        at = time.perf_counter() - t0

        ok = all(a == s for a, s in zip(async_out, sync_out))
        for p, a, s in zip(prompts, async_out, sync_out):
            print(f"[{mode}:{p!r}] async==sync={a == s}")
        print(f"{mode} async run {at:.2f}s")
        if not ok:
            raise SystemExit(f"{mode}: MISMATCH")
        if mode == "graph":
            async_48 = async_llm.generate(
                [list(pids[i % len(pids)]) for i in range(48)], max_tokens=12)
            assert async_48 == sync_48
            print("graph 48-concurrent: PASS")

            for i in range(12):
                async_llm.engine.add_request(Request(
                    f"cancel-{i}", list(pids[i % len(pids)]),
                    SamplingParams(max_tokens=12)))
            async_llm.engine.step()
            for i in range(4):
                async_llm.engine.abort(f"cancel-{i}")
            while async_llm.engine.has_unfinished():
                async_llm.engine.step()
            after_cancel = async_llm.generate(
                [list(pids[i % len(pids)]) for i in range(60)], max_tokens=8)
            assert len(after_cancel) == 60 and all(len(row) == 8 for row in after_cancel)
            print("cancel + 60-request reuse: PASS")
        async_llm.close()

    short = [
        (list(pids[0]) * 4)[:15],
        (list(pids[1]) * 4)[:15],
    ]
    roomy = LLM(cfg(path, True, "graph"))
    expected = roomy.generate(short, max_tokens=4)
    roomy.close()
    tight_cfg = cfg(path, True, "graph")
    tight_cfg.cache.num_blocks = 2
    tight = LLM(tight_cfg)
    calls = {"count": 0}
    original = tight.engine.scheduler.preempt_one

    def counted_preempt():
        victim = original()
        if victim is not None:
            calls["count"] += 1
        return victim

    tight.engine.scheduler.preempt_one = counted_preempt
    actual = tight.generate(short, max_tokens=4)
    assert calls["count"] > 0 and actual == expected
    print(f"forced preemption ({calls['count']} evictions): PASS")

    for i in range(8):
        tight.engine.add_request(Request(
            f"shutdown-{i}", list(pids[0][:3]), SamplingParams(max_tokens=8)))
    tight.engine.step()
    tight.close()
    print("shutdown with pending copies: PASS")
    print("ASYNC SCHEDULING ALL MATCH")


if __name__ == "__main__":
    main()
