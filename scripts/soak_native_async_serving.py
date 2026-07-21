"""Long-running deterministic Serving soak with request accounting."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def soak(args) -> dict:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0.0,
    }
    deadline = time.monotonic() + args.duration_seconds
    submitted = completed = failed = 0
    latencies = []
    reference = None
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"), timeout=600.0,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:
        async def worker():
            nonlocal submitted, completed, failed, reference
            while time.monotonic() < deadline:
                started = time.monotonic()
                async with lock:
                    submitted += 1
                try:
                    response = await client.post(
                        "/v1/completions", json=body, headers=headers
                    )
                    response.raise_for_status()
                    text = response.json()["choices"][0]["text"]
                    async with lock:
                        if reference is None:
                            reference = text
                        if text != reference:
                            failed += 1
                        else:
                            completed += 1
                            latencies.append(time.monotonic() - started)
                except (httpx.HTTPError, KeyError, ValueError):
                    async with lock:
                        failed += 1

        await asyncio.gather(*[worker() for _ in range(args.concurrency)])
        metrics = await client.get("/metrics")

    mean = statistics.mean(latencies) if latencies else 0.0
    cv = (statistics.stdev(latencies) / mean
          if len(latencies) > 1 and mean else 0.0)
    active = submitted - completed - failed
    return {
        "status": "PASS" if failed == 0 and active == 0 else "FAIL",
        "model": args.model,
        "duration_seconds": args.duration_seconds,
        "concurrency": args.concurrency,
        "submitted": submitted,
        "completed": completed,
        "failed": failed,
        "active": active,
        "latency_coefficient_of_variation": cv,
        "metrics_status": metrics.status_code,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain continuous batching briefly.")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--duration-seconds", type=float, default=86400.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(soak(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
