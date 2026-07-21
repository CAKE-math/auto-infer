"""DeepSeek-V2-Lite MLA+MoE NPU greedy smoke (BOS handled).
  python scripts/smoke_deepseek_v2.py /data1/models/DeepSeek-V2-Lite-Chat
"""
import sys
import torch
import torch_npu  # noqa
from transformers import AutoTokenizer
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    dev = npu_device(0)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    m = DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
    for prompt in ["The capital of France is", "2 + 2 =", "Once upon a time"]:
        ids = [tok.bos_token_id] + tok(prompt).input_ids   # DeepSeek add_bos_token=False
        for _ in range(20):
            t = torch.tensor(ids, dtype=torch.long, device=dev)
            pos = torch.arange(len(ids), dtype=torch.long, device=dev)
            nxt = int(m.forward_dense(t, pos)[-1].float().argmax())
            ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
        print(f"[{prompt!r}] -> {tok.decode(ids[1:])!r}")


if __name__ == "__main__":
    main()
