"""Compare a saved online completion with the native offline greedy engine."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                               ModelConfig, SchedulerConfig)
from auto_infer.entrypoints.llm import LLM


def _digest(token_ids: list[int]) -> str:
    payload = ",".join(map(str, token_ids)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--online-response", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-gear", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    prompt_ids = tokenizer.encode(args.prompt)
    config = EngineConfig(
        model=ModelConfig(
            model_path=args.model,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
        ),
        cache=CacheConfig(
            block_size=args.block_size, num_blocks=args.num_blocks
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=16, max_num_batched_tokens=2048
        ),
        execution=ExecutionConfig(
            mode="graph", device_index=0, max_gear=args.max_gear
        ),
    )
    llm = LLM(config)
    try:
        offline_ids = list(llm.generate(
            [list(prompt_ids)], max_tokens=args.output_tokens
        )[0])
    finally:
        llm.close()

    online = json.loads(args.online_response.read_text())
    online_text = online["choices"][0]["text"]
    online_ids = tokenizer.encode(online_text, add_special_tokens=False)
    offline_text = tokenizer.decode(offline_ids)
    result = {
        "status": "PASS" if (
            offline_text == online_text and offline_ids == online_ids
        ) else "FAIL",
        "model": args.model,
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(offline_ids),
        "text_equal": offline_text == online_text,
        "token_ids_equal": offline_ids == online_ids,
        "offline_token_digest": _digest(offline_ids),
        "online_token_digest": _digest(online_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
