# SP4 — Full vLLM-structure async (skeleton refactor 4/5)

Status: Draft · Date: 2026-07-17 · Branch: feat/skeleton-sp4 · Target: npu2

## Context & goal
Two findings drive this:
1. The batch-queue async we built (`EngineCore._step_async`) gives ~0 benefit and is a net
   loss at batch 64 — it defers only the tiny token D2H, while the model forward and the
   D2H both run INLINE on the engine thread. It does NOT match vLLM.
2. SP3 exposed that `PagedNpuExecutor` (async path) decodes at 33 tok/s while a sync eager
   path hits ~476 — the async batch-queue's per-step bookkeeping (submit/collect/clone/
   merge/optimistic-advance + inline `.tolist()` D2H) is itself the cost.

vLLM's uniproc async (read from `vllm/v1/executor/uniproc_executor.py`): `execute_model`
runs the forward on the calling thread but returns an `AsyncModelRunnerOutput` holding GPU
tensors WITHOUT the D2H; `get_output()` (the D2H + CPU output build) is submitted to a
single `async_output_thread` (ThreadPoolExecutor) as a **Future**; the engine thread then
schedules + submits the NEXT batch while that future resolves, blocking only when the queue
is full (`step_with_batch_queue`).

**SP4 makes our async match that structure: the D2H/output-materialization (`collect`'s
`.tolist()`) runs on a dedicated output thread as a future, so the engine thread's
schedule + `_build` + forward-dispatch of step N+1 overlaps the D2H of step N.** Success
criterion: async decode must now **beat** sync decode (today it loses).

## Design
### `worker/model_runner.py` (PagedNpuExecutor / NpuModelRunner) + graph runner
- Add a single-worker `ThreadPoolExecutor` (the output thread) to the executor.
- `submit(...)` unchanged in spirit: runs the forward (eager or graph replay) on the calling
  thread, returns a handle holding the on-device sampled `(B,)` tensor + order (NO D2H).
  `sampled_of` stays (device views, threaded to next batch's decode input — no sync).
- **`collect(handle)` returns a `Future`**: it submits the `.tolist()` D2H + rid-mapping to
  the output thread and returns the future. A new `collect_result(future) -> dict[str,int]`
  blocks on it. (Or: `collect` stays sync and the ENGINE submits the D2H to the thread — pick
  whichever keeps the Executor contract clean; keep `execute()` sync-convenience working.)
- Graph decode path (`GraphPagedNpuExecutor`) must also support this async collect (today it
  is sync-only). Give it the same submit/sampled_of/collect protocol so `supports_async()`
  can return True and it runs through `_step_async` — THIS is where the win compounds (cheap
  graph dispatch on engine thread ∥ D2H on output thread).

### `engine/engine_core.py` `_step_async`
- Keep the depth-`async_batches` queue. Change the queue entries to carry the output-thread
  future; when popping the oldest, block on its future (`collect_result`) — that block now
  overlaps the device compute of the queued newer batches AND the engine already issued
  their schedule/build/dispatch. The invariant (preempt only when queue empty; merge
  `self._sampled`; drop preempted rid) stays exactly as SP-system-mechanics left it.
- No change to scheduler/KV/model/backends.

## Verification (npu2, Qwen2.5-0.5B) — the async-must-win gate
- **Correctness**: full regression (parity, smoke_engine_npu, prefix_cache, preemption,
  graph parity, host suite) — outputs unchanged (async is a timing optimization).
- **Perf (the point)**: extend `bench_async_scheduling.py` / `bench_graph_decode.py` to
  compare async-ON vs async-OFF for BOTH the eager and the graph executor. Gate: async ≥
  sync (must no longer regress; expect a gain on the graph path where dispatch is cheap and
  the D2H/host is the exposed cost). If async still doesn't beat sync, that is a finding —
  report it and DO NOT claim the win; investigate where per-step host time actually goes
  (add a coarse per-phase timer: schedule / build / dispatch / collect).
- Thread-safety: the output thread only reads the handle's device tensor and does `.tolist()`;
  the engine must not mutate that tensor before the future resolves (it's the popped/oldest
  batch — already out of the fill loop). Verify no data race on `self._sampled` (device
  views) between threads.

## Files
- Modify: `worker/model_runner.py`, `worker/graph_decode_runner.py` (async collect),
  `engine/engine_core.py` (`_step_async` future handling), bench scripts.
- Untouched: scheduler, KV manager, models, backends.

## Risks
- Threading + NPU streams: the D2H `.tolist()` on another thread must see the forward's
  result — NPU stream semantics mean the D2H waits on the compute stream; ensure the output
  thread's D2H is ordered after the submit's kernels (same device, default stream ordering
  usually suffices, but verify — a wrong-stream read is silent corruption).
- If threading yields no win on this single-card 0.5B setup (host may still be dominated by
  Python schedule/_build, not the D2H), SP4's honest outcome may be "structure aligned to
  vLLM, marginal single-card gain" — report truthfully rather than forcing a number.
