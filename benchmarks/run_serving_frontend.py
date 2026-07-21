"""Entry point for isolated frontend smoke and live Serving comparisons."""

import argparse
import json
from pathlib import Path

from benchmarks.run_serving_online import main as online_main
from benchmarks.serving_common import serving_result


def _smoke_result() -> dict:
    return serving_result(
        framework="auto-infer-smoke",
        model="dummy",
        git_commit="smoke",
        workload={
            "prompt_tokens": 4,
            "output_tokens": 2,
            "arrival_rate": 0.0,
            "concurrency": 1,
            "warmup_requests": 0,
            "measured_requests": 1,
        },
        ttft_samples=[0.001],
        itl_samples=[0.001],
        e2e_samples=[0.002],
        elapsed_seconds=0.002,
        completed=1,
        rejected=0,
        failed=0,
        client_cpu_seconds=0.001,
        client_peak_rss_bytes=1,
        server_cpu_seconds=0.0,
        server_peak_rss_bytes=1,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    if not args.smoke:
        return online_main([*remaining, "--output", str(args.output)])
    result = _smoke_result()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
