#!/usr/bin/env python3
"""Compare greedy output and serving telemetry between TP1 and TPN endpoints."""

import argparse
import json
import platform
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _get(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode()


def _metrics(base_url: str, timeout_s: float) -> dict[str, float]:
    values = {}
    for line in _get(base_url.rstrip("/") + "/metrics", timeout_s).splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        if name.startswith("auto_infer_serving_prefix_cache_"):
            values[name] = float(value)
    return values


def _generate(
    base_url: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    body = json.dumps({
        "model": "validation",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    text = []
    completion_tokens = 0
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("text", "")
            if delta and first_token_at is None:
                first_token_at = time.perf_counter()
            text.append(delta)
            completion_tokens = max(
                completion_tokens, int(chunk.get("completion_tokens", 0))
            )
    finished = time.perf_counter()
    ttft = None if first_token_at is None else first_token_at - started
    tpot = (
        None
        if ttft is None or completion_tokens <= 1
        else (finished - first_token_at) / (completion_tokens - 1)
    )
    return {
        "text": "".join(text),
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "duration_s": finished - started,
        "throughput_tok_s": (
            completion_tokens / (finished - started)
            if finished > started else None
        ),
    }


def _npu_snapshot() -> str | None:
    executable = shutil.which("npu-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "info"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.stdout + result.stderr


def _run_batch(
    base_url: str,
    prompts: list[str],
    max_tokens: int,
    timeout_s: float,
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(
                _generate, base_url, prompt, max_tokens, timeout_s
            )
            for prompt in prompts
        ]
        return [future.result() for future in futures]


def _prompts(args) -> list[str]:
    prompts = list(args.prompt)
    if args.prompts_file is not None:
        prompts.extend(
            line for line in args.prompts_file.read_text().splitlines()
            if line.strip()
        )
    return prompts or [
        "Explain why deterministic inference matters in one sentence.",
        "用一句话解释张量并行。",
        "The first three prime numbers are",
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    endpoints = {
        "reference": args.reference_url,
        "candidate": args.candidate_url,
    }
    for url in endpoints.values():
        _get(url.rstrip("/") + "/health", args.timeout)
    cases = []
    batch_cases = []
    all_match = True
    before = _npu_snapshot()
    for prompt in _prompts(args):
        results = {
            name: _generate(url, prompt, args.max_tokens, args.timeout)
            for name, url in endpoints.items()
        }
        for result in results.values():
            result["token_ids"] = tokenizer.encode(
                result["text"], add_special_tokens=False
            )
        match = (
            results["reference"]["token_ids"]
            == results["candidate"]["token_ids"]
        )
        all_match &= match
        cases.append({
            "prompt": prompt,
            "token_exact_match": match,
            **results,
        })
    for batch_size in args.batch_size or [4, 16]:
        prompts = [
            f"Tensor parallel continuous batching request {index}:"
            for index in range(batch_size)
        ]
        results = {
            name: _run_batch(
                url, prompts, args.max_tokens, args.timeout
            )
            for name, url in endpoints.items()
        }
        matches = []
        for reference, candidate in zip(
            results["reference"], results["candidate"]
        ):
            reference["token_ids"] = tokenizer.encode(
                reference["text"], add_special_tokens=False
            )
            candidate["token_ids"] = tokenizer.encode(
                candidate["text"], add_special_tokens=False
            )
            matches.append(
                reference["token_ids"] == candidate["token_ids"]
            )
        all_match &= all(matches)
        batch_cases.append({
            "batch_size": batch_size,
            "token_exact_match": all(matches),
            "per_request_match": matches,
            **results,
        })
    prefix_prompt = (
        "AutoInfer tensor parallel prefix-cache validation context. " * 24
        + "Conclude in one sentence:"
    )
    prefix_cases = {
        name: [
            _generate(url, prefix_prompt, args.max_tokens, args.timeout)
            for _ in range(2)
        ]
        for name, url in endpoints.items()
    }
    for results in prefix_cases.values():
        for result in results:
            result["token_ids"] = tokenizer.encode(
                result["text"], add_special_tokens=False
            )
    prefix_match = (
        prefix_cases["reference"][1]["token_ids"]
        == prefix_cases["candidate"][1]["token_ids"]
    )
    all_match &= prefix_match
    artifact = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "npu_smi_before": before,
            "npu_smi_after": _npu_snapshot(),
        },
        "configuration": {
            "reference_url": args.reference_url,
            "candidate_url": args.candidate_url,
            "tokenizer": args.tokenizer,
            "max_tokens": args.max_tokens,
        },
        "token_exact_match": all_match,
        "cases": cases,
        "continuous_batching": batch_cases,
        "prefix_cache_repeat": {
            "token_exact_match": prefix_match,
            **prefix_cases,
        },
        "prefix_cache_metrics": {
            name: _metrics(url, args.timeout)
            for name, url in endpoints.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
