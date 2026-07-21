"""Matched vLLM-Ascend EAGLE3 K-depth benchmark on Qwen3-8B."""
import hashlib
import json
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TARGET = "/data2/models/Qwen3-8B"
DRAFTER = "/data2/models/Qwen3-8B-speculator.eagle3"
TEXTS = (
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "Once upon a time, in a small village at the edge of a forest,",
)


def digest(outputs):
    payload = json.dumps(
        [output.outputs[0].token_ids for output in outputs],
        separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def request_digests(outputs):
    return [
        hashlib.sha256(json.dumps(
            output.outputs[0].token_ids,
            separators=(",", ":")).encode()).hexdigest()[:12]
        for output in outputs
    ]


mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4
if mode not in {"plain", "eagle3"}:
    raise SystemExit("mode must be plain or eagle3")

tokenizer = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
prompts = [TEXTS[index % len(TEXTS)] for index in range(batch_size)]
kwargs = {}
if mode == "eagle3":
    kwargs["speculative_config"] = {
        "model": DRAFTER,
        "method": "eagle3",
        "num_speculative_tokens": depth,
    }
llm = LLM(
    model=TARGET, tensor_parallel_size=1, max_model_len=512,
    max_num_seqs=16, gpu_memory_utilization=0.8,
    trust_remote_code=True, seed=0, disable_log_stats=False, **kwargs)
sampling = SamplingParams(temperature=0, max_tokens=96, ignore_eos=True)
llm.generate(prompts, SamplingParams(
    temperature=0, max_tokens=8, ignore_eos=True))

times = []
outputs = None
for _ in range(3):
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    times.append(time.perf_counter() - started)
tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
median = sorted(times)[1]
print(json.dumps({
    "mode": mode, "K": depth if mode == "eagle3" else 0,
    "batch": batch_size, "tokens": tokens,
    "median_s": median, "tok_s": tokens / median,
    "digest": digest(outputs), "samples_s": times,
    "request_digests": request_digests(outputs),
}, sort_keys=True))
