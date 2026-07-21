# Phase-one architecture acceptance — 2026-07-19

Environment: npu2, Ascend 910B1, CANN 9.0.0 container. Source and logs were
isolated under `/data2/auto-infer-architecture-20260719`; devices 5 and 6 were
left untouched because they were occupied.

| Gate | Device(s) | Result | Retained log |
|---|---:|---|---|
| Linux host suite | CPU | 167 passed | `logs/host-tests.log` |
| Qwen dense/HF first-token parity | 0 | match | `logs/qwen-dense.log` |
| Paged-FIA prefill argmax parity | 0 | all 11 positions match | `logs/qwen-prefill-parity.log` |
| Graph vs eager-FIA decode | 0 | all eight streams token-identical | `logs/qwen-graph.log` |
| Async vs sync scheduling | 0 | all four streams token-identical | `logs/qwen-async.log` |
| Persistent IPC | 1 | 10 streamed tokens, clean close | `logs/qwen-ipc.log` |
| Concurrent serving | 2 | 48/48 complete, 530 tok/s | `logs/qwen-concurrent-serving.log` |
| Cancellation/reuse stress | 0 | one cancellation, 60 later requests complete | `logs/qwen-service-stability.log` |
| Qwen TP2 | 3,4 | output matches single-card dense sequence | `logs/qwen-tp2.log` |
| DeepSeek SP2×EP2 token parity | 0,1,2,7 | 12 generated tokens equal single-card on all ranks | `logs/deepseek-sp2-ep2.log` |
| Config-driven named HCCL mesh | 0,1,2,7 | orthogonal EP sums and SP gathers pass | `logs/config-parallel-mesh.log` |
| Moonlight-16B MLA/MoE paged engine | 0 | coherent 12-token completion | `logs/moonlight-paged.log` |

One historical check, `smoke_paged.py`, reports long-greedy divergence between
paged FIA and a full-softmax dense implementation for two of three prompts.
The prefill argmax gate passes at every tested position, and graph is exactly
equal to eager FIA. The long-run divergence is therefore recorded as a
cross-kernel floating-point difference, not hidden as a pass. It prevents a
claim of universal token identity between different attention kernels.

Phase one is accepted for architecture convergence, the tested correctness
contracts, and persistent-service stability. It does not validate every model,
kernel shape, quantization mode, or physical multi-node topology.
