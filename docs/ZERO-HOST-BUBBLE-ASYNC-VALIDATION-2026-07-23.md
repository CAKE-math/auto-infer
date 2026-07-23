# Zero-Host-Bubble Async Decode Validation

Date: 2026-07-23  
Device: one Ascend 910B1 on `npu2`  
Model: `/data1/models/Qwen3-0.6B`, BF16  
Scope: graph decode, greedy sampling, async depth 2

## Result

The steady-state host bubble has been removed. The final Chrome trace contains
14 comparable decode transitions; all 14 submit the next graph before the
current graph's device ArgMax finishes. Dynamic graph-task updates run on
dedicated host workers and NPU streams after replay submission; every update
finishes before its own graph samples. The device event chain preserves data
ordering without a D2H dependency.

“Zero host time” means that host scheduling and dispatch are absent from the
steady device critical path. It does not mean the CPU performs no work.

## Matched 20-run performance

| Metric | Sync | Async | Delta |
| --- | ---: | ---: | ---: |
| B1 TPOT | 6.071 ms | 4.777 ms | **21.32% lower** |
| B16 throughput | 2,229.45 tok/s | 2,632.16 tok/s | **18.06% higher** |
| B16 throughput CV | 0.94% | 4.81% | async remains below 5% |
| Warm TTFT | 6.550 ms | 39.030 ms | async prefill barrier is slower |
| Output digest | `d23029216ed08f2c` | `d23029216ed08f2c` | identical |

The async implementation is therefore a decode optimization, not a TTFT
optimization. Prefill/mixed execution remains an explicit eager correctness
barrier while async decode is enabled; it must not be described as a prefill
win.

## Trace gates

| Gate | Result |
| --- | ---: |
| `submit_end(N+1) < sample_end(N)` | 14/14 pass |
| graph-task update hidden before its own sample | 14/14 pass |
| sampled-output clone in steady decode | 0 |
| device graph gap p50 | 6.50 μs |
| device graph gap p95 | 6.75 μs |
| p95 gap / async median TPOT | 0.141% |

The trace is directly loadable in Chrome/Perfetto:

- `docs/profiling/qwen3/raw/auto-infer-async.trace.json`
- `docs/profiling/qwen3/auto-infer-async.metadata.json`
- `docs/profiling/qwen3/auto-infer-async-timeline.json`

## Correctness and lifetime gates

The NPU suite matched synchronous output for:

- B1 and B16 greedy decode;
- staggered continuous batching with request joins and exits;
- EOS landing before queued lookahead work;
- cancellation while later batches still hold KV leases.

The combined suite digest is `e1a2b08c3149731b`. EOS requests produced one
token, the cancelled request did not affect its surviving peer, and staggered
requests completed at 17, 11, 7, and 5 tokens.

The implementation enforces the following ownership rules:

1. Each in-flight decode slot owns its captured graph, static tensors, pinned
   staging, graph registrations, update stream, and output lifetime.
2. H2D metadata runs on a slot-specific staging stream.
3. The previous sampled token is spliced after a device event, using persistent
   pinned/device index buffers; the common aligned batch path allocates no
   index tensor.
4. Graph replay is submitted before graph-task updates. Captured external
   events make the graph wait only at the dynamic attention dependency.
5. D2H consumes slot output independently. A request skipped across slot reuse
   is spilled to stable device storage only when needed.
6. EOS, stop, and abort retire a request immediately but reclaim KV and the
   device-token row only after the last submitted batch drops its lease.

## Supported boundary

- Async scheduling is accepted only by executors with isolated in-flight
  slots. The paged/eager runner no longer falsely advertises async safety.
- Async lookahead currently accepts history-independent greedy sampling only.
  Repetition, presence, and frequency penalties are rejected because an
  optimistic placeholder cannot be a valid sampling history.
- MTP remains incompatible with this async mode; its own device pipeline must
  implement the same slot and evidence contract before combination.
- Prefill and mixed batches use the safe barrier described above.

## Stable architecture versus per-model generated state

These must not change per model:

- `prepare -> submit_prepared -> sampled_of -> collect_async` executor contract;
- request retire/reclaim and in-flight KV lease semantics;
- slot, staging-stream, update-worker, event, D2H, and token-spill ownership;
- trace marker names and release gates.

These must be regenerated and revalidated for every model/device combination:

- graph gears and captured graphs;
- graph-task registration handles;
- static buffer shapes, block-table width, and scratch capacity;
- model/backend capability selection;
- BF16 correctness digests, TPOT/throughput samples, CV, and Chrome traces.

## Evidence

Committed artifacts:

- `docs/profiling/qwen3/auto-infer-sync-benchmark.json`
- `docs/profiling/qwen3/auto-infer-async-benchmark.json`
- `docs/profiling/qwen3/auto-infer-async-correctness.json`
- `docs/profiling/qwen3/raw/auto-infer-async.trace.json`

Remote originals:

- `/data2/auto-infer-zero-host-async-20260723/results/`
- `/data2/auto-infer-zero-host-async-20260723/logs/`

Reproduction:

```bash
python scripts/verify_zero_host_async.py /data1/models/Qwen3-0.6B
python benchmarks/run_auto_infer.py benchmarks/qwen3_async_manifest.json
AUTO_INFER_ASYNC_TRACE=1 \
  python benchmarks/profile_qwen3.py \
    benchmarks/qwen3_async_manifest.json auto-infer results/profile
python scripts/analyze_async_timeline.py \
  results/profile/auto-infer.trace.json
```
