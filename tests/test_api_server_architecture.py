import inspect

from auto_infer.serving import api_server


def test_api_server_is_native_async_and_has_no_process_global_runtime():
    source = inspect.getsource(api_server)

    assert "ThreadingHTTPServer" not in source
    assert "BaseHTTPRequestHandler" not in source
    assert "asyncio.to_thread" not in source
    assert not hasattr(api_server, "_TOK")
    assert not hasattr(api_server, "_ENGINE")
    assert not hasattr(api_server, "_MODEL")
