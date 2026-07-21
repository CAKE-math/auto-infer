# Multi-step MTP validation — 2026-07-20

## Result

MiMo-7B now supports configurable recurrent MTP. `K` means the number of draft
tokens proposed before one target verification; `B4` means four concurrent
requests in the batch. The production-signed multi-step configuration is K=2.

| Mode | Batch | Output parity | Throughput | Relative to graph plain |
|---|---:|---:|---:|---:|
| graph plain | B4 | reference | 230.2 tok/s | 1.00x |
| graph MTP K=1 | B4 | 4/4 identical | 290.5 tok/s | 1.27x |
| graph MTP K=2 | B4 | 4/4 identical | 289.9 tok/s | 1.26x |

The K=2 result also exceeds the previously matched vllm-ascend MiMo B4 result
of 176.8 tok/s. K=1 remains the default because this checkpoint has one trained
MTP layer and its recurrent second draft has low acceptance on some prompts;
multi-step is a workload choice, not an unconditional speedup.

## EAGLE3 control

To separate a multi-step execution problem from the quality limit of recurrently
reusing MiMo's single trained MTP layer, vLLM 0.20.2 with vllm-ascend
0.20.2rc1 was tested with the
official `Qwen/Qwen3-8B` target and
`RedHatAI/Qwen3-8B-speculator.eagle3` drafter. `B16` means 16 concurrent
requests; each request emits 96 tokens. Results are medians of three warm runs
on the same Ascend 910B1 device:

| Mode | B4 | B16 | B4 vs K1 | B16 vs K1 |
|---|---:|---:|---:|---:|
| plain | 161.6 tok/s | 605.0 tok/s | 0.86x | 0.86x |
| EAGLE3 K=1 | 187.0 tok/s | 701.0 tok/s | 1.00x | 1.00x |
| EAGLE3 K=3 | 213.4 tok/s | 831.2 tok/s | **1.14x** | **1.19x** |

K=3 therefore accelerates over K=1 on both batch sizes. Its observed
per-position acceptance rates were approximately 63%, 38%, and 12%, producing
about 2.1 output tokens per target verification step. This control shows that
the MiMo K=2 plateau is not caused by multi-step scheduling inherently lacking
speedup; it is dominated by the low marginal acceptance of a second proposal
from a recurrently reused one-step head.

The vllm-ascend EAGLE3 comparison did **not** pass a bitwise greedy parity gate.
Against plain at the same batch size, K=1 matched 3/4 B4 requests and 16/16 B16
requests; K=3 matched 1/4 B4 requests and 7/16 B16 requests. This is consistent
with shape-dependent BF16 numerical paths (plain itself produced different
digests for the same prompts at B4 and B16), but it means these EAGLE3 numbers
are performance evidence, not production precision sign-off. Auto-infer's MiMo
K=1 and K=2 results above retain 4/4 parity against matched graph plain.

## Architecture

- Target verification width is derived from configuration (`K+1`), never from
  hard-coded `K, T` constants.
- The trained MiMo MTP layer is shared by eager and graph paths and is recurrently
  reused. Recurrence consumes the post-final-RMSNorm hidden state.
- The existing two-stage target/drafter graphs remain intact. K=2 adds a
  prewarmed one-token continuation graph and keeps hidden/token hand-off on NPU.
- The scheduler reserves target verification and proposer-continuation KV slots
  separately, while only target query rows consume the scheduler token budget.
- Target logits use the exact fused final-normalization result used by ordinary
  decode; the MTP and target paths share the model's resident-dtype LM head.

## Verification

- Local host suite: 350 tests passed after the final scheduler edge-case fixes.
- npu2 container host suite: 348 tests passed before those host-only edge-case
  fixes; the NPU graph path was revalidated separately.
- NPU: Ascend 910B1 device 1, MiMo-7B-Base, four prompts × 96 output tokens.
- Logs: `/data2/auto-infer-multistep-mtp-20260720/results/`.
- EAGLE3 logs: `/data2/auto-infer-multistep-mtp-20260720/results/eagle3/`.

Re-run the K=1/K=2 production gate with:

```bash
scripts/run_multistep_mtp_validation.sh \
  /data2/auto-infer-multistep-mtp-20260720 1
```
