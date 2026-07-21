# Architecture convergence validation (2026-07-21)

## Result

- Local and npu2-container host suites after the final convergence pass:
  416 passed.
- Moonlight-16B-A3B-Instruct paged-engine smoke: coherent output, passed.
- MiMo graph MTP K1: token-identical to graph-greedy, 295.5 tok/s versus
  231.5 tok/s (1.28x).
- MiMo graph MTP K2: token-identical to graph-greedy, 242.5 tok/s versus
  234.4 tok/s (1.03x). Acceptance by position: 79.8%, 8.59%.

K2 is slower than K1 because the recurrent second proposal runs a complete MTP
layer but is accepted only 8.59% of the time. Persistent staging and a chained
continuation replay remove framework allocations; they cannot turn a proposal
with this marginal acceptance into a throughput gain.

K3 was also tested and is deliberately not exposed: its third-position
acceptance was 3.09% and the four-query-row NPU verification path lost token
identity against stepwise graph-greedy. Correctness therefore caps the current
single-trained-layer recurrent topology at the verified K2 boundary. This is a
capability check derived from the topology and validation result, not a hidden
`K, T` shape constant in the execution path.

## Architectural changes

- Engine-owned admission now enforces model length, greedy-only MTP semantics,
  and the incompatibility between MTP's device pipeline and the async batch
  queue.
- Drafts are one-step leases; missing replacements clear stale state.
- MTP reports acceptance for every proposal position.
- Continuation metadata uses persistent pinned buffers and dirty block rows.
- Engine metrics no longer depend on the serving layer; public errors live in a
  neutral module.
- Distributed runtime construction has one config-driven `ParallelMesh` path.
- `MtpDrafter` uses its normal initializer for an existing head and a
  geometry-aware `from_model` factory; graph gear storage is separated from
  orchestration.
- `BatchPlan` uses fixed-length token views instead of copying the full prompt
  and generated history on every step.
- P/D remains an experimental low-level KV-transfer contract and is not wired
  into serving. MLA MTP has an explicit capability seam but remains unsupported;
  only the validated recurrent GQA path is registered.
