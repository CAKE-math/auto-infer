# Capability and verification matrix

This document separates code presence from current verification. A capability
is marked verified only by an existing test/script and, for NPU behavior, a
retained run from the current architecture acceptance cycle.

## Current architecture acceptance

| Area | Implementation evidence | Current status |
|---|---|---|
| Request admission and no-progress handling | [engine_core.py](../auto_infer/engine/engine_core.py), [test_engine_core.py](../tests/test_engine_core.py) | Host verified |
| Immutable engine/executor boundary | [execution.py](../auto_infer/engine/execution.py), [test_execution_contract.py](../tests/test_execution_contract.py) | Host verified |
| One runtime configuration and executor factory | [factory.py](../auto_infer/engine/factory.py), [test_executor_factory.py](../tests/test_executor_factory.py) | Host verified |
| Persistent service and request broker | [service.py](../auto_infer/serving/service.py), [test_serving_engine.py](../tests/test_serving_engine.py) | Host verified |
| Concurrent IPC demultiplexing | [ipc.py](../auto_infer/serving/ipc.py), [test_serving_ipc.py](../tests/test_serving_ipc.py) | Host verified |
| Multi-replica routing | Deliberately outside the single-engine serving scope | Not integrated |
| Named TP/DP/EP/CP/SP mesh | [mesh.py](../auto_infer/distributed/mesh.py), [test_parallel_mesh.py](../tests/test_parallel_mesh.py) | Host and four-rank HCCL verified |
| Model-independent attention selection | [registry.py](../auto_infer/layers/attention/registry.py), [test_graph_decode_runner.py](../tests/test_graph_decode_runner.py) | Host plus GQA/MLA NPU paths verified |
| Public API and CLI | [cli.py](../auto_infer/entrypoints/cli.py), [test_documentation_integrity.py](../tests/test_documentation_integrity.py) | Host verified |

## NPU paths to revalidate

| Path | Reproduction entrypoint | Acceptance requirement |
|---|---|---|
| Qwen dense and paged | [smoke_qwen2.py](../scripts/smoke_qwen2.py), [smoke_paged.py](../scripts/smoke_paged.py) | coherent output and first-token/paged parity |
| Qwen graph decode | [smoke_graph_engine.py](../scripts/smoke_graph_engine.py) | graph output equals eager-FIA output |
| DeepSeek MLA paged | `scripts/verify_deepseek_graphdecode.py` | unified paged/graph output passes the retained NPU parity gate |
| Moonlight V3-style MLA/MoE | [verify_v3_paged_moonlight.py](../scripts/verify_v3_paged_moonlight.py) | persistent paged engine completes coherently |
| Async scheduling | [verify_async_sched.py](../scripts/verify_async_sched.py) | token identity with synchronous execution |
| IPC and concurrent serving | [verify_ipc_serving.py](../scripts/verify_ipc_serving.py), [verify_async_serving.py](../scripts/verify_async_serving.py) | persistent process/service completes concurrent requests |
| TP | [smoke_tp_qwen2.py](../scripts/smoke_tp_qwen2.py) | two-rank output agrees with single-rank reference |
| EP/SP | [verify_sp_ep_exact.py](../scripts/verify_sp_ep_exact.py) | distributed result agrees with reference |

Current-cycle NPU logs are retained under
`/data2/auto-infer-architecture-20260719/logs` on npu2. See the
[phase-one report](PHASE1-VALIDATION-2026-07-19.md) and
[three-framework comparison](ARCHITECTURE-COMPARISON.md).

## Explicit limitations

- DeepSeek-V3 671B multi-node execution is not validated without the checkpoint,
  sufficient nodes, and working cross-node fabric.
- Mooncake/RDMA PD transport is optional and is not considered verified merely
  because its adapter imports.
- MTP numerical validation requires a compatible checkpoint containing trained
  MTP weights.
- Kernel support depends on CANN/torch_npu versions and model dimensions; tiny
  random checkpoints are useful for architecture loading but may not satisfy FIA
  kernel shape constraints.
- The controlled phase-two result does not support universal performance
  superiority; see the comparison report for the split outcome.
