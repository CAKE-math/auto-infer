# Harness NPU Validation — 2026-07-24

Environment: `npu2`, Ascend 910B1, container
`auto-infer-dev-20260624`. The isolated source, generated packages, and raw
results are retained under:

```text
/data2/auto-infer-harness-20260724/
```

## Results

| Gate | Result |
|---|---|
| Container host suite | 516 passed, 14 upstream deprecation warnings |
| Directed Harness suite | 17 passed |
| Qwen2.5-0.5B inspect/adapt/validate | passed |
| Qwen2.5 architecture-alias BF16 token parity | exact match |
| Qwen3-8B architecture-alias BF16 token parity | exact match |

The final Qwen3 package registered the previously unknown architecture
`HarnessQwen3ForCausalLM` with
`auto_infer.models.qwen3:Qwen3Model`. Two greedy requests, eight generated
tokens each, matched the built-in Qwen3 path exactly:

```text
reference digest = dc4a1aef0d1626a4173c91d62acbd4780e9d0eb00c7ed91aa088c98631d3e15c
candidate digest = dc4a1aef0d1626a4173c91d62acbd4780e9d0eb00c7ed91aa088c98631d3e15c
mismatched requests = []
```

Key retained artifacts:

- `results/qwen3-alias-adapt-fixed.stdout.json`
- `results/qwen3-alias-validate-fixed.stdout.json`
- `results/qwen3-alias-token-parity-fixed.json`
- `packages/qwen3-alias/model-package.json`
- `results/host-tests.log`

## Defects found by real-checkpoint validation

1. Qwen2.5 declares a window size while explicitly setting
   `use_sliding_window=false`. Inspection originally treated the size alone as
   enabled and incorrectly returned `partial`. The inspector now honors the
   explicit enable switch, with all three legacy/true/false cases tested.
2. Qwen3-8B has an explicit head dimension equal to its derived dimension.
   Template selection originally classified it as Qwen2. The matcher now uses
   Q/K Norm weight evidence as the decisive Qwen3 capability signal, preventing
   omission of Q/K normalization.

Both corrections were made test-first and the final Qwen3 package was
regenerated before the passing parity result above.
