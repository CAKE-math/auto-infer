"""NPU smoke + HF parity for the auto-infer Qwen2 model.

Run inside the NPU container:
  python scripts/smoke_qwen2.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.models.qwen2 import Qwen2Model
from auto_infer.platform import npu_device


def greedy(model, tok, prompt, n=20):
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    dev = model.device
    for _ in range(n):
        t = torch.tensor(ids, dtype=torch.long, device=dev)
        pos = torch.arange(len(ids), dtype=torch.long, device=dev)
        logits = model.forward_dense(t, pos)
        nxt = int(logits[-1].float().argmax().item())
        ids.append(nxt)
        if nxt == tok.eos_token_id:
            break
    return ids


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    dev = npu_device(0)
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)

    prompt = "The capital of France is"
    ids = greedy(model, tok, prompt, n=16)
    text = tok.decode(ids)
    print("=== GREEDY OUTPUT ===")
    print(repr(text))

    # HF parity on first-token argmax (CPU reference)
    try:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        in_ids = tok(prompt, return_tensors="pt").input_ids
        hf_logits = hf(in_ids).logits[0, -1]
        hf_tok = int(hf_logits.argmax().item())

        t = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
        pos = torch.arange(t.shape[0], dtype=torch.long, device=dev)
        our_tok = int(model.forward_dense(t, pos)[-1].float().argmax().item())
        print("=== PARITY ===")
        print(f"HF next-token  = {hf_tok} ({tok.decode([hf_tok])!r})")
        print(f"our next-token = {our_tok} ({tok.decode([our_tok])!r})")
        print("MATCH" if hf_tok == our_tok else "MISMATCH")
    except Exception as e:  # transformers model load optional
        print(f"[parity skipped: {e}]")


if __name__ == "__main__":
    main()
