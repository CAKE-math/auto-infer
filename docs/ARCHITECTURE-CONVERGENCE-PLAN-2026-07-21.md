# Architecture convergence plan (2026-07-21)

## Design

The engine core owns admission invariants. A request must be executable by the
selected backend before it reaches the scheduler: model length is bounded,
speculative decoding is explicitly greedy-only, and speculative decoding is
not silently combined with the separate async batch queue.

Speculative execution has one source of truth for proposal depth and reports
acceptance by proposal position. Missing next-step drafts clear carried state.
All continuation metadata is staged through persistent host/device buffers;
the graph runner orchestrates replay instead of constructing Python lists and
device tensors in its hot loop.

Infrastructure shared by the engine and serving layers lives below both. The
engine stat logger and public exceptions therefore move to neutral modules.
Distributed initialization keeps one config-driven `ParallelMesh` path plus a
clearly isolated environment compatibility path; compatibility code is not
deleted without topology tests.

## Execution plan

1. Add failing engine-contract tests for model length, speculative sampling,
   async/spec incompatibility, and stale-draft clearing; implement centrally.
2. Add per-position speculative acceptance statistics and logger tests.
3. Add host-testable continuation staging with persistent buffers and dirty-row
   block-table updates; integrate it into ACL graph replay.
4. Replace manual `MtpDrafter.__new__` assembly with an explicit constructor,
   remove live dead wrappers/parameters, and split graph-MTP support code where
   it reduces orchestration density.
5. Move engine metrics and exceptions to neutral layers; preserve compatibility
   imports and simplify distributed group planning under tests.
6. Refresh contracts/docstrings, run static checks and all host tests.
7. Deploy the resulting tree to npu2 `/data2`, run correctness/stability smoke
   tests and compare MTP1/MTP2 throughput before accepting the hot-path change.

## Acceptance

- No accepted request can exceed `max_model_len` or silently use unsupported
  speculative sampling semantics.
- Speculative state cannot survive a step that produced no replacement draft.
- Continuation replay performs no per-step `torch.tensor(..., device=npu)`
  construction and updates unchanged block-table rows zero times.
- Full host suite passes and NPU output matches the same graph-greedy numeric
  regime. Proposal depths that fail identity validation are rejected rather
  than advertised as supported.
