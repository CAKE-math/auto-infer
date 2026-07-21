"""DEBUG: per-layer HF parity bisect for DeepSeek-V2-Lite. Finds the FIRST layer
where our forward(ctx)+MlaFIABackend diverges from HF transformers (reference),
to localize the coherent-output bug (our DeepSeek was only parity-checked vs our
own legacy, never vs HF).

  python tools/parity_hf_deepseek.py /data1/models/DeepSeek-V2-Lite-Chat
"""
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.platform import npu_device
from auto_infer.forward_context import ForwardContext

BLOCK = 16


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/data1/models/DeepSeek-V2-Lite-Chat"
    dev = npu_device(0)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
    P = len(ids)
    nb = (P + BLOCK - 1) // BLOCK

    # ---- ours (forward(ctx) + MlaFIABackend, paged prefill) with per-layer capture ----
    ours = DeepseekV2Model.from_pretrained(path, device=dev, dtype=torch.bfloat16)
    token_ids = torch.tensor(ids, dtype=torch.long, device=dev)
    positions = torch.arange(P, dtype=torch.long, device=dev)
    slot_mapping = torch.arange(P, dtype=torch.int32, device=dev)
    block_table = torch.arange(nb, dtype=torch.int32, device=dev).view(1, nb)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=dev), diagonal=1)
    be, kv = ours.make_attention_backend(nb, BLOCK)
    ctx = ForwardContext(token_ids=token_ids, positions=positions, slot_mapping=slot_mapping,
                         block_table=block_table, cu_seqlens_q=[P], seqlens_kv=[P],
                         attn_mask=mask, attn_backend=be, kv_caches=kv, is_decode=False)
    ours._dbg = []
    our_embed = ours.w["model.embed_tokens.weight"][token_ids].detach().float().cpu()
    our_logits = ours.logits(ours.forward(ctx)).float().cpu()
    our_layers = ours._dbg                                   # [after layer 0, 1, ...]
    del ours, kv; torch.npu.empty_cache()

    # ---- HF reference ----
    # DeepSeek's trust_remote_code modeling imports a symbol newer transformers
    # dropped; shim it (it's just a torch.fx feature-gate) so the import works.
    import transformers.utils.import_utils as _iu
    for _sym in ("is_torch_fx_available", "is_torch_fx_proxy"):
        if not hasattr(_iu, _sym):
            setattr(_iu, _sym, (lambda *a, **k: False))
    # newer transformers dropped DynamicCache.from_legacy_cache that the bundled
    # DeepSeek modeling calls — shim it to return an empty cache (fine for one
    # cache-less forward).
    try:
        from transformers.cache_utils import DynamicCache as _DC
        if not hasattr(_DC, "from_legacy_cache"):
            _DC.from_legacy_cache = classmethod(lambda cls, past=None: cls())
    except Exception:
        pass
    try:
        hf = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16).to(dev).eval()
        hdev = dev
    except Exception as e:
        print(f"[HF on NPU failed: {e}; falling back to CPU fp32]")
        hf = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                  torch_dtype=torch.float32).eval()
        hdev = torch.device("cpu")
    with torch.no_grad():
        out = hf(torch.tensor([ids], device=hdev), output_hidden_states=True, use_cache=False)
    hf_hs = [h[0].float().cpu() for h in out.hidden_states]   # [embed, after l0, after l1, ...]
    hf_logits = out.logits[0].float().cpu()

    print(f"=== HF PARITY BISECT (DeepSeek-V2-Lite, P={P}, layers={len(our_layers)}) ===")
    print(f"embed max|Δ| = {(our_embed - hf_hs[0]).abs().max():.4g}")
    for i, oh in enumerate(our_layers):
        d = (oh - hf_hs[i + 1]).abs().max().item()
        flag = "  <== FIRST DIVERGENCE" if d > 0.5 and (i == 0 or (our_layers[i-1] - hf_hs[i]).abs().max() <= 0.5) else ""
        print(f"  after layer {i:2d}: max|Δ| = {d:.4g}{flag}")
    print(f"final next-token: ours={our_logits[-1].argmax().item()} hf={hf_logits[-1].argmax().item()} "
          f"({'MATCH' if our_logits[-1].argmax()==hf_logits[-1].argmax() else 'MISMATCH'})")


if __name__ == "__main__":
    main()
