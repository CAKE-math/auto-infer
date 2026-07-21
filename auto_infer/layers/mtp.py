import torch

from auto_infer.layers.mlp import swiglu_mlp
from auto_infer.layers.norm import add_rms_norm, rms_norm
from auto_infer.forward_context import ForwardContext


class RecurrentMtpHead:
    """One trained MiMo MTP layer shared by eager and graph executors."""

    def __init__(self, model, backend, kv_caches, attn_mask, prefix):
        self.model = model
        self.backend = backend
        self.kv_caches = kv_caches
        self.attn_mask = attn_mask
        self.prefix = prefix

    def hidden(self, hidden, token_ids, positions, slots, block_table,
               cu_seqlens_q, seqlens_kv):
        w, p, eps = self.model.w, self.prefix, self.model.cfg.rms_eps
        combined = torch.cat(
            [rms_norm(hidden, w[p + "hidden_layernorm.weight"], eps),
             rms_norm(w["model.embed_tokens.weight"][token_ids],
                      w[p + "token_layernorm.weight"], eps)], -1)
        combined = combined @ w[p + "input_proj.weight"].t()
        ctx = ForwardContext(
            token_ids=None, positions=positions, slot_mapping=slots,
            block_table=block_table, cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=seqlens_kv, attn_mask=self.attn_mask,
            attn_backend=self.backend, kv_caches=self.kv_caches,
            is_decode=False)
        cos, sin = self.model._compute_cos_sin(positions)
        ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        residual = combined
        normalized = rms_norm(
            combined, w[p + "input_layernorm.weight"], eps)
        attended = self.backend.attention(0, normalized, ctx)
        activated, residual = add_rms_norm(
            attended, residual,
            w[p + "post_attention_layernorm.weight"], eps)
        output = residual + swiglu_mlp(activated, w, p + "mlp.")
        return rms_norm(output, w[p + "final_layernorm.weight"], eps)

    def forward(self, *args):
        hidden = self.hidden(*args)
        return hidden, self.model.logits(hidden)
