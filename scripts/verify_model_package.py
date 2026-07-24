"""Compare a generated model package with a trusted auto-infer model.

Run this on the target NPU after ``inferctl validate package``. The candidate
model may use an architecture alias supplied by the package; the reference
model uses the built-in registry.
"""

import argparse
import hashlib
import json

from auto_infer.config import (
    CacheConfig,
    EngineConfig,
    ExecutionConfig,
    ModelConfig,
    SchedulerConfig,
)
from auto_infer.entrypoints.llm import LLM


def output_digest(outputs: list[list[int]]) -> str:
    payload = json.dumps(
        outputs, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_outputs(reference: list[list[int]],
                    candidate: list[list[int]]) -> dict:
    count = max(len(reference), len(candidate))
    mismatches = [
        index for index in range(count)
        if index >= len(reference)
        or index >= len(candidate)
        or reference[index] != candidate[index]
    ]
    return {
        "exact_match": not mismatches,
        "mismatched_requests": mismatches,
        "reference_digest": output_digest(reference),
        "candidate_digest": output_digest(candidate),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify token identity for an auto-infer model package")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--prompt-ids", required=True,
                        help="JSON list of token-id lists")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--mode", default="paged")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=4096)
    return parser


def _run(model_path: str, prompts: list[list[int]], args,
         package: str | None = None) -> list[list[int]]:
    config = EngineConfig(
        model=ModelConfig(
            model_path=model_path,
            model_package=package,
        ),
        cache=CacheConfig(num_blocks=args.num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096),
        execution=ExecutionConfig(
            mode=args.mode,
            device_index=args.device,
        ),
    )
    llm = LLM(config)
    try:
        return llm.generate(prompts, max_tokens=args.max_tokens)
    finally:
        llm.close()


def _release_device_cache() -> None:
    try:
        import torch
        npu = getattr(torch, "npu", None)
        if npu is not None:
            npu.empty_cache()
    except (ImportError, RuntimeError):
        pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    prompts = json.loads(args.prompt_ids)
    if (not isinstance(prompts, list) or not prompts
            or any(not isinstance(prompt, list) or not prompt
                   or any(not isinstance(token, int) for token in prompt)
                   for prompt in prompts)):
        raise ValueError("--prompt-ids must be a non-empty JSON list of "
                         "non-empty integer lists")

    reference = _run(args.reference_model, prompts, args)
    _release_device_cache()
    candidate = _run(
        args.candidate_model, prompts, args, package=args.package)
    comparison = compare_outputs(reference, candidate)
    result = {
        "status": "ok" if comparison["exact_match"] else "failed",
        "comparison": comparison,
        "reference_outputs": reference,
        "candidate_outputs": candidate,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if comparison["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
