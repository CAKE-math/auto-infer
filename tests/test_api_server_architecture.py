import inspect
import sys
from types import SimpleNamespace

from auto_infer.serving import api_server
from auto_infer.serving.config import ServingConfig


def test_api_server_is_native_async_and_has_no_process_global_runtime():
    source = inspect.getsource(api_server)

    assert "ThreadingHTTPServer" not in source
    assert "BaseHTTPRequestHandler" not in source
    assert "asyncio.to_thread" not in source
    assert not hasattr(api_server, "_TOK")
    assert not hasattr(api_server, "_ENGINE")
    assert not hasattr(api_server, "_MODEL")


def test_build_runtime_uses_exact_provided_engine():
    tokenizer = object()
    engine = object()
    config = ServingConfig()

    runtime = api_server.build_runtime(
        tokenizer=tokenizer,
        model="test-model",
        max_model_len=2048,
        serving_config=config,
        engine=engine,
    )

    assert runtime.tokenizer is tokenizer
    assert runtime.engine is engine
    assert runtime.model == "test-model"
    assert runtime.max_model_len == 2048
    assert runtime.serving_config is config


def test_run_runtime_builds_app_and_delegates_to_uvicorn(monkeypatch):
    runtime = object()
    app = object()
    calls = []
    monkeypatch.setattr(api_server, "create_app", lambda value: (
        calls.append(("create_app", value)) or app
    ))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
        run=lambda value, **kwargs: calls.append(("run", value, kwargs))
    ))

    api_server.run_runtime(
        runtime, host="127.0.0.1", port=8123, access_log=True
    )

    assert calls == [
        ("create_app", runtime),
        ("run", app, {
            "host": "127.0.0.1",
            "port": 8123,
            "access_log": True,
        }),
    ]
