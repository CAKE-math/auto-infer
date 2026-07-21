"""NPU verification for prefix caching (spec §1/§5b).

Runs the paged async engine on a shared prompt: after one request finishes and
registers its blocks, an identical second request must MATCH the cached prefix
(match_prefix returns blocks) and skip recomputing it, while producing the SAME
tokens as a cold-cache baseline.

  python scripts/verify_prefix_cache.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys

from transformers import AutoTokenizer

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.entrypoints.llm import LLM
from auto_infer.worker.model_runner import PagedNpuExecutor

BLOCK_SIZE = 16
NUM_BLOCKS = 512


def build_llm(path):
    cfg = EngineConfig(
        model=ModelConfig(model_path=path),
        cache=CacheConfig(block_size=BLOCK_SIZE, num_blocks=NUM_BLOCKS),
        scheduler=SchedulerConfig(max_num_batched_tokens=2048,
                                  enable_prefix_caching=True),
    )
    return LLM(cfg, executor=PagedNpuExecutor(path, NUM_BLOCKS, BLOCK_SIZE))


def instrument_match(llm):
    """Wrap the engine's KVCacheManager.match_prefix to record the max number of
    tokens matched (0 = no hit). Returns a mutable dict updated on each call."""
    kv = llm.engine.kv
    stat = {"max_matched_blocks": 0, "calls": 0}
    orig = kv.match_prefix

    def wrapped(token_ids):
        blocks = orig(token_ids)
        stat["calls"] += 1
        stat["max_matched_blocks"] = max(stat["max_matched_blocks"], len(blocks))
        return blocks

    kv.match_prefix = wrapped
    return stat


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    # A shared prefix long enough that, even after the (num_prompt_tokens-1) match
    # cap, it still spans several full BLOCK_SIZE blocks — otherwise a 1-block
    # prompt caps to 0 matchable blocks and the cache can never hit.
    prompt = ("You are a helpful assistant that answers questions clearly and "
              "concisely for the user. Use complete sentences and stay factual. "
              "Here is the question you must answer now: what is the capital "
              "city of France, and what is it famous for around the world?")
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    assert (len(ids) - 1) // BLOCK_SIZE >= 2, (
        f"prompt too short ({len(ids)} tok) to span >=2 blocks at block_size "
        f"{BLOCK_SIZE}; lengthen it to exercise the prefix cache")
    n_gen = 16

    # cold-cache baseline (fresh engine, never seen this prompt)
    base_out = build_llm(path).generate([list(ids)], max_tokens=n_gen)[0]

    # cache run: warm the cache by finishing an identical request, then re-run it
    llm = build_llm(path)
    llm.generate([list(ids)], max_tokens=n_gen)          # registers blocks on free
    stat = instrument_match(llm)
    cached_out = llm.generate([list(ids)], max_tokens=n_gen)[0]

    matched_tokens = stat["max_matched_blocks"] * BLOCK_SIZE
    print("=== PREFIX CACHE VERIFY ===")
    print(f"prompt tokens          = {len(ids)}")
    print(f"match_prefix calls     = {stat['calls']}")
    print(f"max matched blocks     = {stat['max_matched_blocks']} "
          f"({matched_tokens} tokens, cap = {(len(ids) - 1) // BLOCK_SIZE} blocks)")
    print(f"baseline out           = {base_out}")
    print(f"cached  out            = {cached_out}")
    ok_hit = stat["max_matched_blocks"] > 0
    ok_eq = cached_out == base_out
    print(f"prefix HIT             : {'PASS' if ok_hit else 'FAIL'}")
    print(f"output == baseline     : {'PASS' if ok_eq else 'FAIL'}")
    print("RESULT:", "PASS" if (ok_hit and ok_eq) else "FAIL")
    sys.exit(0 if (ok_hit and ok_eq) else 1)


if __name__ == "__main__":
    main()
