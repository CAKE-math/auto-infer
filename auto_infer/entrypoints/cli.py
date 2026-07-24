"""Command-line entrypoint for supported auto-infer workflows."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-infer")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the OpenAI-compatible server")
    serve.add_argument("model")
    serve.add_argument("--model-package")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", type=int, default=0)
    serve.add_argument(
        "--mode", choices=("recompute", "paged", "graph", "graph_mtp"),
        default="paged")
    serve.add_argument("--max-model-len", type=int, default=4096)
    serve.add_argument("--num-blocks", type=int, default=4096)
    serve.add_argument("--block-size", type=int, default=16)
    serve.add_argument("--max-num-seqs", type=int, default=256)
    serve.add_argument("--max-num-batched-tokens", type=int, default=8192)
    serve.add_argument("--max-gear", type=int, default=32)
    serve.add_argument("--max-prefill-tokens", type=int, default=256)
    serve.add_argument("--num-speculative-tokens", type=int, default=1)
    serve.add_argument("--api-key")
    serve.add_argument("--max-http-inflight", type=int, default=512)
    serve.add_argument("--max-waiting-requests", type=int)
    serve.add_argument("--max-waiting-tokens", type=int, default=1_048_576)
    serve.add_argument("--tokenizer-batch-size", type=int, default=32)
    serve.add_argument("--tokenizer-queue-capacity", type=int, default=1024)
    serve.add_argument("--tokenizer-wait-ms", type=float, default=2.0)
    serve.add_argument("--sse-coalesce-ms", type=float, default=5.0)
    serve.add_argument("--sse-coalesce-tokens", type=int, default=8)
    serve.add_argument("--shutdown-grace-s", type=float, default=30.0)
    serve.add_argument(
        "--access-log", action="store_true",
        help="enable per-request uvicorn access logging (off by default)",
    )
    return parser


def _serving_config(args):
    from auto_infer.serving.config import ServingConfig

    return ServingConfig(
        max_num_seqs=args.max_num_seqs,
        max_http_inflight=args.max_http_inflight,
        max_waiting_requests=args.max_waiting_requests,
        max_waiting_tokens=args.max_waiting_tokens,
        tokenizer_batch_size=args.tokenizer_batch_size,
        tokenizer_queue_capacity=args.tokenizer_queue_capacity,
        tokenizer_wait_ms=args.tokenizer_wait_ms,
        sse_coalesce_ms=args.sse_coalesce_ms,
        sse_coalesce_tokens=args.sse_coalesce_tokens,
        shutdown_grace_s=args.shutdown_grace_s,
        api_key=args.api_key,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        from auto_infer.serving.api_server import serve
        serve(model_path=args.model, host=args.host, port=args.port,
              model_package=args.model_package,
              device_index=args.device, mode=args.mode,
              max_model_len=args.max_model_len, num_blocks=args.num_blocks,
              block_size=args.block_size, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_gear=args.max_gear,
              max_prefill_tokens=args.max_prefill_tokens,
              num_speculative_tokens=args.num_speculative_tokens,
              access_log=args.access_log,
              serving_config=_serving_config(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
