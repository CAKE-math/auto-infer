import urllib.request, json, sys, traceback
port = sys.argv[1] if len(sys.argv) > 1 else "8000"
try:
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions",
        data=json.dumps({"prompt": "The capital of France is", "max_tokens": 12}).encode(),
        headers={"Content-Type": "application/json"})
    print("RESP:", urllib.request.urlopen(req, timeout=180).read().decode(), flush=True)
except Exception as e:
    print("CLIENT_ERR:", repr(e), flush=True); traceback.print_exc()
