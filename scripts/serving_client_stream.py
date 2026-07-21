import urllib.request, json, sys
port=sys.argv[1] if len(sys.argv)>1 else "8000"
req=urllib.request.Request(f"http://localhost:{port}/v1/completions",
    data=json.dumps({"prompt":"The capital of France is","max_tokens":10,"stream":True}).encode(),
    headers={"Content-Type":"application/json"})
n=0; text=""
for line in urllib.request.urlopen(req, timeout=180):
    line=line.decode().strip()
    if line.startswith("data: ") and "[DONE]" not in line:
        n+=1; text+=json.loads(line[6:])["choices"][0]["text"]
print(f"STREAM chunks={n} text={text!r}")
