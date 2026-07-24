# Production TP Serving Validation — 2026-07-24

## Verdict

BF16 dense-Qwen tensor-parallel serving is operational in `recompute` and
`paged` modes on one node with 2–8 Ascend NPUs. Qwen2.5-72B loaded by rank on
eight 910B cards, exposed the production HTTP API, completed B1/B4/B16
continuous batches, reused prefix-cache blocks, and shut down as one failure
domain.

`graph` and `graph_mtp` TP are deliberately rejected before process creation.
The graph TP service reached `/health`, but its first generation did not
complete. A healthy control plane is not sufficient evidence for graph replay
liveness, so the capability remains closed.

One accuracy observation remains open: in two paged Qwen3-8B B16 trials, one
TP1 response differed from its own sequential result while the TP2 response
matched the sequential result. A later B16 run was exact. This does not show a
TP result error, but it does show that a single TP1 batched run is not a stable
oracle for adversarial near-tie prompts. Production accuracy qualification
must therefore use repeated trials and an independent reference, not only one
TP1-versus-TPN comparison.

## Source and environment

- Source commit tested after the final host gate: `9d93643`
- Remote root: `/data2/auto-infer-tp-serving-20260724`
- Host: `npu2` (`liteserver-c007-8-0`)
- Device: 8 × Ascend 910B1, 64 GiB HBM each
- Container: `auto-infer-dev-20260624`
- Python: 3.11.15
- PyTorch: 2.10.0
- torch-npu: 2.10.0
- npu-smi: 25.5.0
- Host gate, local: `574 passed`
- Host gate, Ascend container: `574 passed`

The source was copied to the remote root without checkpoints or build caches.
Raw logs and JSON artifacts are retained below that root in `logs/` and
`artifacts/`.

## Qwen3-8B TP1 versus TP2

Model: `/data2/models/Qwen3-8B`

The paged TP2 replica used devices 1 and 2. Rank-local HBM was approximately
13.0 GiB per rank, versus approximately 28.0 GiB for TP1. This is consistent
with slice-at-read for transformer projections; no TP rank materialized the
full transformer checkpoint.

The first exact comparison generated 32 tokens for three fixed English/Chinese
prompts:

| Gate | Result |
|---|---:|
| Sequential greedy token equality, 3 × 32 tokens | 3/3 exact |
| B4 continuous batch, 16 tokens/request | 4/4 exact |
| Repeated long prefix output | exact |
| Prefix queried / hit blocks | 30 / 15 |
| Prefix hit rate | 0.5 |

The B16 caveat described in the verdict affected one TP1 response in two
trials. The TP2 response equaled the prompt's sequential TP1 output. A later
run with a different arrival grouping produced 16/16 equality. The retained
artifacts are:

- `artifacts/tp1-vs-tp2-qwen3-8b.json`
- `artifacts/tp1-vs-tp2-qwen3-8b-full.json`
- `artifacts/tp1-vs-tp2-qwen3-8b-b16-rerun.json`
- `artifacts/tp1-tokenizer32-vs-tokenizer1-b16.json`

## Qwen2.5-72B TP8

Model: `/data1/models/Qwen2.5-72B-Instruct`

The initial deployment smoke below has now been followed by a matched
three-framework online-serving comparison against vllm-ascend and Omni-NPU.
See
[`QWEN25-72B-TP8-THREE-FRAMEWORK-COMPARISON-2026-07-24.md`](QWEN25-72B-TP8-THREE-FRAMEWORK-COMPARISON-2026-07-24.md).
The larger-model result does not preserve auto-infer's earlier single-card
Qwen3 ranking: the current paged-eager TP path trails both graph-enabled
baselines in steady decode throughput.

The eight-rank paged replica reached `/health` and completed the production
`/v1/completions` path. Per-rank process HBM was 24.5–25.7 GiB. All eight ranks
were active.

| Workload | Aggregate wall time | Completion |
|---|---:|---:|
| B1 × 16 tokens | 5.735 s | 1/1 complete |
| B4 × 16 tokens | 3.241 s | 4/4 complete |
| B16 × 16 tokens | 3.928 s | 16/16 complete |

The B1 result includes first-request kernel and collective warm-up. B16
delivered 256 tokens in 3.928 seconds, or approximately 65.2 aggregate
tokens/second. A repeated long-prefix request was text-identical and reported:

- queried blocks: 24
- hit blocks: 12
- hit rate: 0.5

Raw result: `artifacts/qwen25-72b-tp8-smoke.json`.

## Replica failure gate

Killing the TP2 follower initially exposed a supervisor defect: SIGTERM let
rank0 enter graceful shutdown and wait for an ACK from the dead follower. The
supervisor now uses a bounded two-stage teardown:

1. send `terminate()` to every live child;
2. wait briefly;
3. send `kill()` to any survivor;
4. join every process with a bound.

After the fix, killing rank1 removed the complete replica and released both
NPUs in 3 seconds. Raw result:
`artifacts/tp2-follower-kill-fixed.txt`.

## Capability boundary

Generated dense-GQA model packages declare:

```json
{
  "tensor": {
    "status": "supported",
    "dtype": "bfloat16",
    "max_size": 8,
    "modes": ["recompute", "paged"]
  },
  "expert": {"status": "unsupported"}
}
```

MLA/MoE packages declare tensor parallel unsupported and BF16 expert parallel
supported. Quantization remains disabled with its interface reserved. TP MTP
and TP graph modes fail before allocating models or KV caches.

## Reproduction

Start a TP8 replica:

```bash
python -m auto_infer.entrypoints.cli serve \
  /data1/models/Qwen2.5-72B-Instruct \
  --tp-size 8 --devices 0,1,2,3,4,5,6,7 \
  --master-port 29601 --tp-watchdog-timeout 600 \
  --host 127.0.0.1 --port 18400 --mode paged \
  --max-model-len 4096 --num-blocks 2048 --max-num-seqs 32
```

Compare two production endpoints:

```bash
python scripts/validate_tp_serving.py \
  --reference-url http://127.0.0.1:18100 \
  --candidate-url http://127.0.0.1:18200 \
  --tokenizer /data2/models/Qwen3-8B \
  --max-tokens 32 \
  --output artifacts/tp-validation.json
```

The validator records output token IDs derived with the checkpoint tokenizer,
text, TTFT, TPOT, aggregate throughput, B4/B16 continuous batching,
prefix-cache counters, and `npu-smi` snapshots. It exits non-zero on any token
mismatch.
