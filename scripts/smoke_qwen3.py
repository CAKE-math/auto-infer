"""NPU smoke + HF parity for the auto-infer Qwen3 dense model (QK-Norm, no attn
bias, independent head_dim — all data-driven in the shared GQA backend).

Run inside the NPU container:
  python scripts/smoke_qwen3.py /data1/models/Qwen3-0.6B
"""
import json
import os
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.models.registry import get_model_class
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
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/Qwen3-0.6B"
    dev = npu_device(0)
    with open(os.path.join(path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    tok = AutoTokenizer.from_pretrained(path)
    model = get_model_class(arch).from_pretrained(path, device=dev, dtype=torch.bfloat16)
    print(f"arch={arch} -> {type(model).__name__}")
    for layer in range(model.cfg.num_layers):
        p = f"model.layers.{layer}."
        assert p + "self_attn.qkv_proj.weight" in model.w
        assert p + "mlp.gate_up_proj.weight" in model.w
        assert p + "self_attn.q_proj.weight" not in model.w
        assert p + "mlp.gate_proj.weight" not in model.w
    print("packed projections: PASS")

    prompt = "The capital of France is"
    ids = greedy(model, tok, prompt, n=16)
    print("=== GREEDY OUTPUT ===")
    print(repr(tok.decode(ids)))

    try:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        in_ids = tok(prompt, return_tensors="pt").input_ids
        hf_tok = int(hf(in_ids).logits[0, -1].argmax().item())

        t = in_ids[0].to(dev)
        pos = torch.arange(t.shape[0], dtype=torch.long, device=dev)
        our_tok = int(model.forward_dense(t, pos)[-1].float().argmax().item())
        print("=== PARITY ===")
        print(f"HF next-token  = {hf_tok} ({tok.decode([hf_tok])!r})")
        print(f"our next-token = {our_tok} ({tok.decode([our_tok])!r})")
        print("MATCH" if hf_tok == our_tok else "MISMATCH")
    except Exception as e:
        print(f"[parity skipped: {e}]")


if __name__ == "__main__":
    main()
