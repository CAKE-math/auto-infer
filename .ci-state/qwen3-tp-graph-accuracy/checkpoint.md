status: completed

# PIE accuracy-regression checkpoint

- axis: mode
- model: Qwen3-8B
- good: TP2 paged BF16
- bad: TP2 ACL Graph BF16
- hardware: npu2, Ascend 910B
- deterministic environment:
  - `HCCL_DETERMINISTIC=true`
  - `LCCL_DETERMINISTIC=1`
  - `CLOSE_MATMUL_K_SHIFT=1`
- sampling: greedy (`temperature=0`)
- reproducer: `用一句话解释张量并行。`, 29 output tokens
- graph probes: authorized by the user's explicit request to use PIE to fix
  the precision issue; temporary in-source probes are gated by
  `PIE_ACCURACY_CAPTURE_DIR`.

## Completed

- TP2 graph first-request liveness passed.
- TP2 graph B4 continuous batching passed.
- Initial non-deterministic graph/paged comparison found late token divergence.
- Deterministic rerun still reproduced:
  - one sequential request diverged at token 28;
  - one of sixteen continuous-batch requests diverged at token 3.
- Confirmed the installed `torch_npu` exposes `save_npugraph_tensor`.

## Next

- Continue the TP8 Qwen2.5-72B performance and stability gates with the
  precision fix enabled.

## Localization result

- PIE mode-axis probes captured embedding, every layer's post-attention and
  post-MLP outputs, final norm, and lm-head logits.
- The first above-noise hidden-state difference was
  `layer8.attn.o_proj.out` (`max_abs=0.001953125`,
  `mean_abs=6.70e-05`, cosine approximately 1.0). This is expected BF16
  accumulation-order drift, not a structural graph error.
- At the first divergent token, paged BF16 logits scored candidates
  101884/23031 as 28.625/28.5; graph quantized both to 28.375 and ordinary
  argmax selected the lower token ID.
- FP32 recomputation over only those candidate weight rows selected token
  101884 on both modes. Full-head FP32 storage was unnecessary.

## Fix and gates

- Added a shared stable-greedy epilogue: BF16 full-vocabulary projection,
  top-candidate selection, transient FP32 candidate dot products, deterministic
  tie handling.
- The epilogue is captured inside ACL Graph and shared by paged, graph prefill,
  graph decode, and target MTP verification.
- Local: 585 tests passed.
- NPU graph capture/startup: passed.
- Deterministic TP2 graph versus paged:
  - sequential: 3/3 token exact;
  - B4: 4/4 token exact;
  - B16: 16/16 token exact;
  - repeated prefix: 2/2 token exact.
- Qwen3 TP2 graph B16 remained approximately 683 tok/s versus approximately
  697 tok/s before the precision epilogue.

## Artifacts

- Remote captures:
  `/data2/auto-infer-tp-graph-20260724/captures/{paged,graph}`
- PIE reports: `/tmp/pie-qwen3.qcNTQo/diff-report-logits.json`
- Fixed graph result:
  `/data2/auto-infer-tp-graph-20260724/results/qwen3-tp2-graph-stable-greedy.json`
- Fixed paged result:
  `/data2/auto-infer-tp-graph-20260724/results/qwen3-tp2-paged-stable-greedy.json`
