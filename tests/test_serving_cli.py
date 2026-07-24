from auto_infer.entrypoints.cli import _serving_config, build_parser


def test_serve_cli_exposes_native_async_limits():
    args = build_parser().parse_args([
        "serve", "/model",
        "--api-key", "secret",
        "--max-http-inflight", "12",
        "--max-waiting-requests", "7",
        "--max-waiting-tokens", "900",
        "--tokenizer-batch-size", "4",
        "--tokenizer-queue-capacity", "20",
        "--tokenizer-wait-ms", "1.5",
        "--sse-coalesce-ms", "3.0",
        "--sse-coalesce-tokens", "6",
        "--shutdown-grace-s", "9.0",
    ])

    config = _serving_config(args)

    assert config.api_key == "secret"
    assert config.max_http_inflight == 12
    assert config.max_waiting_requests == 7
    assert config.max_waiting_tokens == 900
    assert config.tokenizer_batch_size == 4
    assert config.tokenizer_queue_capacity == 20
    assert config.tokenizer_wait_ms == 1.5
    assert config.sse_coalesce_ms == 3.0
    assert config.sse_coalesce_tokens == 6
    assert config.shutdown_grace_s == 9.0
    assert args.access_log is False


def test_serve_cli_can_enable_access_log_explicitly():
    args = build_parser().parse_args(["serve", "/model", "--access-log"])

    assert args.access_log is True


def test_serve_cli_exposes_independent_graph_limits():
    args = build_parser().parse_args([
        "serve", "/model", "--max-gear", "16",
        "--max-prefill-tokens", "192",
    ])

    assert args.max_gear == 16
    assert args.max_prefill_tokens == 192


def test_serve_cli_accepts_explicit_model_package():
    args = build_parser().parse_args([
        "serve", "/model", "--model-package", "/packages/example",
    ])

    assert args.model_package == "/packages/example"
