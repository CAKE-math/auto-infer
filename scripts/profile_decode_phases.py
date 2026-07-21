"""Break open the graph decode per-step cost (v2): the ~8.6 ms/step is all inside
executor.execute() — time its sub-phases (per-layer graph_task_update loop, logits,
and the marshal+replay+sample+D2H remainder).

  python scripts/profile_decode_phases.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys
import time

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.graph_decode_runner import GraphPagedNpuExecutor

BLOCK, NB, N = 16, 4096, 200


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    pid = tok("Explain how a transformer works.", return_tensors="pt").input_ids[0].tolist()
    cfg = EngineConfig(model=ModelConfig(model_path=path),
                       cache=CacheConfig(block_size=BLOCK, num_blocks=NB),
                       scheduler=SchedulerConfig(max_num_batched_tokens=8192))
    llm = LLM(cfg, executor=GraphPagedNpuExecutor(path, NB, BLOCK, max_gear=32))
    runner = llm.engine.executor.runner
    acc = {"execute": 0.0, "update": 0.0, "logits": 0.0, "n": 0}

    def wrap(obj, name, key, sync=False):
        orig = getattr(obj, name)
        def w(*a, **k):
            if sync: torch.npu.synchronize()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            if sync: torch.npu.synchronize()
            acc[key] += time.perf_counter() - t0
            return r
        setattr(obj, name, w)

    llm.generate([list(pid)], max_tokens=8)                      # warmup + capture
    wrap(llm.engine.executor, "execute", "execute", sync=True)   # total per step (device-bounded)
    wrap(runner.backend, "update", "update")                     # 24x graph_task_update host loop
    wrap(runner.model, "logits", "logits")                       # fp32 (B,vocab) projection
    t0 = time.perf_counter(); torch.npu.synchronize()
    outs = llm.generate([list(pid)], max_tokens=N)
    torch.npu.synchronize(); total = time.perf_counter() - t0
    steps = len(outs[0])
    print(f"decode steps={steps}  {steps/total:.1f} tok/s  execute={acc['execute']*1e3/steps:.3f} ms/step")
    for k in ("update", "logits"):
        print(f"  {k:8s} {acc[k]*1e3/steps:7.3f} ms/step  ({100*acc[k]/acc['execute']:4.1f}% of execute)")
    rem = acc["execute"] - acc["update"] - acc["logits"]
    print(f"  {'remainder':8s} {rem*1e3/steps:7.3f} ms/step  ({100*rem/acc['execute']:4.1f}% of execute)  [marshal+replay+sample+D2H]")


if __name__ == "__main__":
    main()
