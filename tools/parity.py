"""Qwen2 ongoing parity check: two different attention implementations
(`GqaFIABackend` vs `DenseBackend`) driven through the SAME unified
`forward(ctx)` on the SAME model/weights must agree on the argmax token at
every position.

Paged FIA and full-softmax dense differ in fp accumulation order, so we do NOT
assert bitwise/near-bitwise equality — the hard gate is argmax agreement;
max|Δ logits| is printed informationally only. (Qwen2's pre-refactor
`forward_paged` was removed once this gate replaced it; DeepSeek keeps its
DeepSeek parity is covered by the unified model/backend verification scripts.)
Run on NPU (uses paged FIA + reshape_and_cache):

  python tools/parity.py /data0/models/Qwen2.5-0.5B-Instruct
"""
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from auto_infer.layers.attention.gqa import GqaFIABackend
from auto_infer.models.qwen2 import Qwen2Model
from auto_infer.platform import npu_device
from auto_infer.forward_context import ForwardContext

BLOCK_SIZE = 16


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data0/models/Qwen2.5-0.5B-Instruct"
    dev = npu_device(0)
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
    cfg = model.cfg

    ids = tok("The capital of France is Paris, a city famous for", return_tensors="pt").input_ids[0].tolist()
    P = len(ids)
    nb = (P + BLOCK_SIZE - 1) // BLOCK_SIZE
    token_ids = torch.tensor(ids, dtype=torch.long, device=dev)
    positions = torch.arange(P, dtype=torch.long, device=dev)
    slot_mapping = torch.arange(P, dtype=torch.int32, device=dev)          # contiguous blocks 0..nb-1
    block_table = torch.arange(nb, dtype=torch.int32, device=dev).view(1, nb)
    cu_q, kvlen = [P], [P]
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=dev), diagonal=1)

    # Path A: unified forward(ctx) + GqaFIABackend (the live serving path).
    be = GqaFIABackend(n_q_heads=model.n_q_local, n_kv_heads=model.n_kv_local,
                         head_dim=cfg.head_dim, scale=cfg.head_dim ** -0.5,
                         num_layers=cfg.num_layers, device=dev, dtype=model.dtype,
                         w=model.w, layer_prefix=model.layer_prefix)
    kv_paged = be.alloc_kv_caches(nb, BLOCK_SIZE)
    ctx_paged = ForwardContext(token_ids=token_ids, positions=positions, slot_mapping=slot_mapping,
                               block_table=block_table, cu_seqlens_q=cu_q, seqlens_kv=kvlen,
                               attn_mask=mask, attn_backend=be, kv_caches=kv_paged, is_decode=False)
    logits_paged = model.logits(model.forward(ctx_paged))                  # (P, vocab)

    # Path B: unified forward(ctx) + DenseBackend, via the forward_dense bring-up helper.
    logits_dense = model.forward_dense(token_ids, positions)               # (P, vocab)

    d = (logits_paged.float() - logits_dense.float()).abs()
    max_abs = float(d.max())
    argmax_match = bool(torch.equal(logits_paged.argmax(-1), logits_dense.argmax(-1)))
    print("=== PARITY: forward(ctx)+GqaFIABackend vs forward(ctx)+DenseBackend ===")
    print(f"seq_len={P}  layers={cfg.num_layers}")
    print(f"max|Δ logits| = {max_abs:.6g}  (informational only — FIA vs full-softmax fp accumulation differs)")
    print(f"argmax match (all {P} positions) = {argmax_match}")
    ok = argmax_match  # hard gate: argmax agreement only, NOT bitwise equality
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
