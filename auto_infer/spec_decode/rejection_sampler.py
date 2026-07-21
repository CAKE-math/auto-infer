"""Branchless greedy rejection sampler.

Given k draft tokens for a sequence and the target model's greedy predictions at
the k+1 verify positions, accept the LONGEST prefix of drafts that the target
would itself have produced, then emit one correction/bonus token. The emitted
tokens are always p[0..m] where p[i]=argmax(target_logits[i]) and m = length of
the leading run of (draft[i]==p[i]); since that is by construction a prefix of
the plain-greedy continuation, spec-decode greedy output is TOKEN-IDENTICAL to
plain autoregressive greedy — independent of draft quality.

Branchless / fixed-shape (graph-capturable, no host control flow):
  matches      = (draft == p[:, :k])           # (B, k) bool
  run          = cumprod(matches, dim=1)        # 1...1 0...0 ; first 0 ends the run
  num_accepted = run.sum(dim=1)                 # (B,) tensor — accepted draft count m
  valid_mask   = positions <= num_accepted      # (B, k+1) ; first m+1 emitted tokens valid
This is the "固定 max + mask" form the spec calls for; the caller appends the
masked-valid emitted tokens and advances KV/position by (m+1).
"""
import torch


def verify_and_accept(draft_tokens: torch.Tensor, target_preds: torch.Tensor):
    """draft_tokens: (B, k) int — the proposed tokens d1..dk.
    target_preds:    (B, k+1) int — argmax of target logits at the k+1 verify
                     positions (p0..pk); p_i is the target's greedy token after
                     verify position i.
    Returns (num_accepted (B,) long, emitted (B, k+1) int == target_preds,
             valid_mask (B, k+1) bool). Emitted[b, :num_accepted[b]+1] are the
             real new tokens for sequence b this step.
    """
    B, k = draft_tokens.shape
    matches = (draft_tokens == target_preds[:, :k])              # (B, k)
    run = torch.cumprod(matches.to(torch.int32), dim=1)          # leading-run indicator
    num_accepted = run.sum(dim=1).to(torch.long)                 # (B,) in [0, k]
    pos = torch.arange(k + 1, device=draft_tokens.device).unsqueeze(0)  # (1, k+1)
    valid_mask = pos <= num_accepted.unsqueeze(1)                # first m+1 valid
    return num_accepted, target_preds, valid_mask
