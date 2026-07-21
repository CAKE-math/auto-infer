"""Independent Transformers greedy-reference probe for MiMo output tokens."""
import hashlib
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/MiMo-7B-Base"
device = sys.argv[2] if len(sys.argv) > 2 else "npu:0"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
prompts = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
    "The history of the Roman Empire began when",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "Once upon a time, in a small village at the edge of a forest,",
]
outputs = []
with torch.no_grad():
    for text in prompts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        outputs.append(generated[0, input_ids.shape[1]:].tolist())

print("GREEDY_REFERENCE " + json.dumps({
    "framework": "transformers",
    "output_digest": hashlib.sha256(
        json.dumps(outputs).encode()).hexdigest()[:16],
    "output_lengths": list(map(len, outputs)),
    "output_token_ids": outputs,
}, sort_keys=True))
