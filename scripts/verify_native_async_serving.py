"""Correctness gate for a running native-async auto-infer text server."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from benchmarks.sse_client import parse_sse_line


async def _stream_text(client, body, headers):
    text = ""
    terminal = False
    async with client.stream(
        "POST", "/v1/completions",
        json={**body, "stream": True}, headers=headers,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            parsed = parse_sse_line(line)
            if parsed is None:
                continue
            delta, done = parsed
            text += delta
            terminal |= done
    return text, terminal


async def verify(args) -> dict:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0.0,
    }
    checks = {}
    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"), timeout=600.0
    ) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models", headers=headers)
        plain = await client.post(
            "/v1/completions", json=body, headers=headers
        )
        stream_text, terminal = await _stream_text(client, body, headers)
        plain.raise_for_status()
        expected = plain.json()["choices"][0]["text"]
        checks["health"] = health.status_code == 200
        checks["model"] = (
            models.status_code == 200
            and any(item["id"] == args.model for item in models.json()["data"])
        )
        checks["stream_matches_nonstream"] = terminal and stream_text == expected

        for batch in (1, 4, 16):
            responses = await asyncio.gather(*[
                client.post("/v1/completions", json=body, headers=headers)
                for _ in range(batch)
            ])
            texts = [response.json()["choices"][0]["text"]
                     for response in responses if response.status_code == 200]
            checks[f"b{batch}_complete"] = len(texts) == batch
            checks[f"b{batch}_greedy_equal"] = texts == [expected] * batch

        invalid = await client.post(
            "/v1/completions",
            json={**body, "max_tokens": 0}, headers=headers,
        )
        checks["invalid_is_400"] = invalid.status_code == 400

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "url": args.url,
        "model": args.model,
        "checks": checks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain continuous batching briefly.")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(verify(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
