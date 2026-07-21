"""Capability/correctness probe for vLLM-compatible MiMo MTP runtimes."""
import hashlib
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/MiMo-7B-Base"
framework = os.environ.get("MTP_FRAMEWORK", "vllm-ascend")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
base_prompts = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "Once upon a time, in a small village at the edge of a forest,",
]
batch_size = int(os.environ.get("MTP_BATCH_SIZE", "4"))
prompts = [base_prompts[index % len(base_prompts)] for index in range(batch_size)]
prompt_ids = [tokenizer(text).input_ids for text in prompts]

load_start = time.perf_counter()
llm = LLM(
    model=path,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=512,
    max_num_seqs=16,
    max_num_batched_tokens=2048,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    enforce_eager=os.environ.get("MTP_ENFORCE_EAGER", "0") == "1",
    seed=0,
    speculative_config={"method": "mtp", "num_speculative_tokens": 1},
)
load_seconds = time.perf_counter() - load_start
params = SamplingParams(
    max_tokens=32, temperature=0.0, ignore_eos=True, seed=0)
request_prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
llm.generate(request_prompts, params, use_tqdm=False)
elapsed_samples = []
outputs = None
sample_count = int(os.environ.get("MTP_SAMPLES", "5"))
for _ in range(sample_count):
    start = time.perf_counter()
    outputs = llm.generate(request_prompts, params, use_tqdm=False)
    elapsed_samples.append(time.perf_counter() - start)
assert outputs is not None
token_ids = [output.outputs[0].token_ids for output in outputs]
digest = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:16]
median_elapsed = statistics.median(elapsed_samples)
mean_elapsed = statistics.mean(elapsed_samples)
print("MTP_PROBE " + json.dumps({
    "framework": framework,
    "batch_size": batch_size,
    "load_seconds": load_seconds,
    "elapsed_samples_seconds": elapsed_samples,
    "median_elapsed_seconds": median_elapsed,
    "elapsed_cv_percent": statistics.pstdev(elapsed_samples) / mean_elapsed * 100.0,
    "throughput_tokens_per_second": sum(map(len, token_ids)) / median_elapsed,
    "output_digest": digest,
    "output_lengths": list(map(len, token_ids)),
    "output_token_ids": token_ids,
}, sort_keys=True))
