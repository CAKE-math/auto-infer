"""NPU verification for recompute preemption + drain-then-preempt (spec §4/§5b).

Runs the paged async engine under deliberate KV pressure (num_blocks sized to fit
only ~one request's full footprint) so running requests must be preempted and
recomputed. Asserts the generated tokens are IDENTICAL to a roomy (no-preemption)
run — i.e. recompute preserves the token stream — and that preemption actually fired.

  python scripts/verify_preemption.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import math
import sys

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

BLOCK_SIZE = 16
N_GEN = 48
ROOMY_BLOCKS = 512


def build_llm(path, num_blocks):
    cfg = EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=2048),
    )
    return LLM(cfg, executor=PagedNpuExecutor(path, num_blocks, BLOCK_SIZE))


def count_preemptions(llm):
    calls = {"n": 0}
    orig = llm.engine.scheduler.preempt_one

    def counted():
        r = orig()
        if r:
            calls["n"] += 1
        return r

    llm.engine.scheduler.preempt_one = counted
    return calls


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    prompts = ["The capital of France is", "The largest planet is"]
    ids = [tok(p, return_tensors="pt").input_ids[0].tolist() for p in prompts]

    # tight = just over ONE request's full footprint, so one can complete but two
    # cannot coexist -> decode-time KV pressure forces preemption.
    single_full = max(math.ceil((len(i) + N_GEN) / BLOCK_SIZE) for i in ids)
    tight = single_full + 1
    print(f"per-request full footprint = {single_full} blocks; tight = {tight} "
          f"blocks; two-request demand ~= {sum(math.ceil((len(i)+N_GEN)/BLOCK_SIZE) for i in ids)} blocks")

    roomy = build_llm(path, ROOMY_BLOCKS).generate([list(i) for i in ids], max_tokens=N_GEN)

    tight_llm = build_llm(path, tight)
    calls = count_preemptions(tight_llm)
    tight_out = tight_llm.generate([list(i) for i in ids], max_tokens=N_GEN)

    print("=== PREEMPTION VERIFY ===")
    print(f"preemptions fired      = {calls['n']}")
    for k, (r, t) in enumerate(zip(roomy, tight_out)):
        print(f"req{k}: match={r == t} (len {len(r)}/{len(t)})")
    ok_fired = calls["n"] > 0
    ok_eq = tight_out == roomy
    print(f"preemption fired       : {'PASS' if ok_fired else 'FAIL'}")
    print(f"tight == roomy stream  : {'PASS' if ok_eq else 'FAIL'}")
    print("RESULT:", "PASS" if (ok_fired and ok_eq) else "FAIL")
    sys.exit(0 if (ok_fired and ok_eq) else 1)


if __name__ == "__main__":
    main()
