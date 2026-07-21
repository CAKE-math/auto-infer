"""Measure an OpenAI-shaped streaming text endpoint with raw latency samples."""

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from benchmarks.serving_common import serving_result
from benchmarks.sse_client import parse_sse_event


@dataclass(frozen=True)
class RequestSample:
    status: str
    ttft: float = 0.0
    itls: tuple[float, ...] = ()
    e2e: float = 0.0


def _request_body(prompt: str, output_tokens: int,
                  model: str | None) -> dict:
    body = {
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    if model:
        body["model"] = model
    return body


async def _request(client: httpx.AsyncClient, *, prompt: str,
                   output_tokens: int, model: str | None,
                   api_key: str | None) -> RequestSample:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    body = _request_body(prompt, output_tokens, model)
    started = time.perf_counter()
    first_at = None
    previous_at = None
    itls = []
    previous_count = 0
    try:
        async with client.stream(
            "POST", "/v1/completions", json=body, headers=headers
        ) as response:
            if response.status_code == 429:
                return RequestSample("rejected")
            if response.status_code != 200:
                await response.aread()
                return RequestSample("failed")
            async for line in response.aiter_lines():
                event = parse_sse_event(line)
                if event is None:
                    continue
                text, terminal = event.text, event.done
                now = time.perf_counter()
                token_arrived = bool(text)
                if event.completion_tokens is not None:
                    token_delta = event.completion_tokens - previous_count
                    previous_count = event.completion_tokens
                    if token_delta > 1:
                        return RequestSample("coalesced")
                    token_arrived = token_delta == 1
                if token_arrived:
                    if first_at is None:
                        first_at = now
                    elif previous_at is not None:
                        itls.append(now - previous_at)
                    previous_at = now
                if terminal:
                    break
    except (httpx.HTTPError, asyncio.TimeoutError):
        return RequestSample("failed")
    ended = time.perf_counter()
    if first_at is None:
        return RequestSample("failed")
    if not itls:
        itls.append(max(0.0, ended - first_at))
    return RequestSample(
        "completed", first_at - started, tuple(itls), ended - started
    )


def _git_revision() -> str:
    explicit_revision = os.getenv("AUTO_INFER_GIT_COMMIT")
    if explicit_revision:
        return explicit_revision
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _server_usage(pid: int) -> tuple[float, int]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        raise RuntimeError(f"cannot sample server process {pid}")
    fields = stat[stat.rfind(")") + 2:].split()
    clock_ticks = os.sysconf("SC_CLK_TCK")
    cpu_seconds = (int(fields[11]) + int(fields[12])) / clock_ticks
    rss_kib = next(
        int(line.split()[1]) for line in status if line.startswith("VmRSS:")
    )
    return cpu_seconds, rss_kib * 1024


async def run_workload(*, url: str, framework: str, model: str,
                       prompt: str, prompt_tokens: int, output_tokens: int,
                       concurrency: int, requests: int, warmup: int,
                       arrival_rate: float, api_key: str | None = None,
                       server_pid: int) -> dict:
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(
        base_url=url.rstrip("/"), limits=limits, timeout=timeout
    ) as client:
        for _ in range(warmup):
            await _request(
                client, prompt=prompt, output_tokens=output_tokens,
                model=model, api_key=api_key,
            )

        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request():
            async with semaphore:
                return await _request(
                    client, prompt=prompt, output_tokens=output_tokens,
                    model=model, api_key=api_key,
                )

        server_started = _server_usage(server_pid)
        peak_server_rss = [server_started[1]]
        sampler_stop = threading.Event()

        def sample_server_rss():
            while not sampler_stop.wait(0.01):
                try:
                    peak_server_rss[0] = max(
                        peak_server_rss[0], _server_usage(server_pid)[1]
                    )
                except RuntimeError:
                    return

        sampler = threading.Thread(
            target=sample_server_rss, name="ServingServerRssSampler",
            daemon=True,
        )
        sampler.start()
        cpu_started = time.process_time()
        started = time.perf_counter()
        tasks = []
        for index in range(requests):
            target = started + index / arrival_rate if arrival_rate > 0 else started
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(bounded_request()))
        samples = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        sampler_stop.set()
        sampler.join(timeout=1)
        server_ended = _server_usage(server_pid)

    completed = [sample for sample in samples if sample.status == "completed"]
    rejected = sum(sample.status == "rejected" for sample in samples)
    failed = sum(sample.status == "failed" for sample in samples)
    coalesced = sum(sample.status == "coalesced" for sample in samples)
    failed += coalesced
    if not completed:
        raise RuntimeError("benchmark completed no successful requests")
    return serving_result(
        framework=framework,
        model=model,
        git_commit=_git_revision(),
        workload={
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "arrival_rate": arrival_rate,
            "concurrency": concurrency,
            "warmup_requests": warmup,
            "measured_requests": requests,
        },
        ttft_samples=[sample.ttft for sample in completed],
        itl_samples=[value for sample in completed for value in sample.itls],
        e2e_samples=[sample.e2e for sample in completed],
        elapsed_seconds=elapsed,
        completed=len(completed),
        rejected=rejected,
        failed=failed,
        client_cpu_seconds=cpu_seconds,
        client_peak_rss_bytes=_peak_rss_bytes(),
        server_cpu_seconds=server_ended[0] - server_started[0],
        server_peak_rss_bytes=max(peak_server_rss[0], server_ended[1]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain continuous batching briefly.")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--api-key", default=os.getenv("AUTO_INFER_API_KEY"))
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(run_workload(
        url=args.url,
        framework=args.framework,
        model=args.model,
        prompt=args.prompt,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        concurrency=args.concurrency,
        requests=args.requests,
        warmup=args.warmup,
        arrival_rate=args.arrival_rate,
        api_key=args.api_key,
        server_pid=args.server_pid,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
