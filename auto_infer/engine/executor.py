from concurrent.futures import Future

import torch

from auto_infer.engine.execution import BatchPlan, DeviceTokenBatch, ExecutionResult


class PreparedExecution:
    """Default prepare/submit split for backends without separate staging."""

    def __init__(self, plan, prev_sampled):
        self.plan = plan
        self.prev_sampled = prev_sampled


class Executor:
    """Abstract execution backend.

    Four-call interface for the vLLM-aligned batch-queue engine:
      submit(sched, scheduler, prev_sampled) -> handle
          Enqueue device work WITHOUT a host sync. `prev_sampled` maps rid ->
          the previous step's sampled token (device tensor / int) used as the
          decode input for this step (no CPU round-trip). Returns an opaque handle.
      sampled_of(handle) -> DeviceTokenBatch | None
          Retains one complete device tensor plus immutable row metadata for the
          NEXT batch's input splice — read WITHOUT syncing or per-row clones.
      collect_async(handle) -> Future[{rid: int}]
          Submit the CPU-materialization (the D2H sync point) to a dedicated
          output thread (matching vLLM's uniproc `async_output_thread`) and
          return a Future immediately, WITHOUT blocking the calling thread.
      collect_result(future) -> {rid: int}
          Block on the future and return the materialized token dict — the
          actual sync point, moved to wherever the caller chooses to await it.
      collect(handle) -> {rid: int}
          Synchronous convenience = collect_result(collect_async(handle)).
          Used by execute() and by callers that don't need overlap.
    execute() is the synchronous convenience path (submit+collect)."""

    @property
    def recoverable(self) -> bool:
        return False

    def close(self) -> None:
        """Release resources owned by the executor."""

    def supports_async(self) -> bool:
        """True if this backend threads sampled tokens (submit/sampled_of/collect) for
        the batch-queue async path. Sync-only backends (recompute) return False and the
        engine drives them synchronously."""
        return False

    def submit(self, plan: BatchPlan, prev_sampled=None):
        raise NotImplementedError

    def prepare(self, plan: BatchPlan, prev_sampled=None):
        return PreparedExecution(plan, prev_sampled)

    def submit_prepared(self, prepared):
        return self.submit(prepared.plan, prepared.prev_sampled)

    def sampled_of(self, handle) -> DeviceTokenBatch | None:
        sampled = handle.get("sampled", {}) if handle else {}
        if not sampled:
            return None
        order = tuple(sampled)
        tokens = torch.tensor([sampled[rid] for rid in order], dtype=torch.long)
        return DeviceTokenBatch.from_output(tokens, order)

    def collect(self, handle) -> ExecutionResult:
        return ExecutionResult.from_single_tokens(handle.get("sampled", {}) if handle else {})

    def collect_async(self, handle) -> Future:
        """Default: run `collect` inline (on the calling thread) and wrap the
        result in an already-resolved Future — no real background thread, so
        this is threadless and deterministic (host tests / MockExecutor rely
        on this). Real async backends (PagedNpuExecutor, GraphPagedNpuExecutor)
        override this to submit the D2H to a single-worker output thread."""
        fut: Future = Future()
        try:
            fut.set_result(self.collect(handle))
        except Exception as exc:                # pragma: no cover - defensive
            fut.set_exception(exc)
        return fut

    def collect_result(self, future: Future) -> ExecutionResult:
        return future.result()

    def release_submission(self, handle) -> None:
        """Release per-submission buffers after its host result is consumed."""

    def release_requests(self, request_ids) -> None:
        """Release runner-owned request state after the engine reclaims KV."""

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        return self.collect(self.submit(plan, {}))

    def execute_spec_mtp(self, plan: BatchPlan) -> ExecutionResult:
        """One MTP speculative-decode super-step (only backends with a trained MTP
        head implement this; EngineCore calls it when a SpecDecodeConfig is set).
        Returns emitted target tokens, the configured-depth next drafts, and
        aggregate plus per-position acceptance statistics."""
        raise NotImplementedError


class RunnerExecutor(Executor):
    """Stable executor boundary for a runner that implements the full protocol."""

    def __init__(self, runner):
        self.runner = runner

    def supports_async(self) -> bool:
        supports = getattr(self.runner, "supports_async", None)
        return bool(supports and supports())

    def prepare(self, plan: BatchPlan, prev_sampled=None):
        return self.runner.prepare(plan, prev_sampled)

    def submit_prepared(self, prepared):
        return self.runner.submit_prepared(prepared)

    def submit(self, plan: BatchPlan, prev_sampled=None):
        return self.runner.submit(plan, prev_sampled)

    def sampled_of(self, handle) -> DeviceTokenBatch | None:
        return self.runner.sampled_of(handle)

    def collect(self, handle) -> ExecutionResult:
        return self.runner.collect(handle)

    def collect_async(self, handle):
        return self.runner.collect_async(handle)

    def collect_result(self, future) -> ExecutionResult:
        return self.runner.collect_result(future)

    def release_submission(self, handle) -> None:
        release = getattr(self.runner, "release_submission", None)
        if release is not None:
            release(handle)

    def release_requests(self, request_ids) -> None:
        release = getattr(self.runner, "release_requests", None)
        if release is not None:
            release(request_ids)

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        return self.runner.execute(plan)

    def execute_spec_mtp(self, plan: BatchPlan) -> ExecutionResult:
        return self.runner.execute_spec_mtp(plan)

    def close(self) -> None:
        self.runner.close()


class MockExecutor(Executor):
    """Deterministic stand-in: samples (last_input_token + 1) % vocab_size once a
    request's prompt is processed. Decode input is the previous sample (threaded via
    prev_sampled), matching the real device-token feed. No model/NPU needed."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self._sampled: dict = {}

    def supports_async(self) -> bool:
        return True

    def execute(self, plan):
        handle = self.submit(plan, self._sampled)
        result = self.collect(handle)
        batch = self.sampled_of(handle)
        if batch is not None:
            self._sampled.update(batch.refs())
        return result

    def submit(self, plan, prev_sampled=None):
        prev_sampled = prev_sampled or {}
        sampled: dict[str, int] = {}
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            start = req.num_computed_tokens                 # pre-advance state
            if start + sr.num_tokens_to_compute >= req.num_prefill_tokens:
                if start >= req.num_prefill_tokens:
                    ref = prev_sampled[sr.request_id]
                    last = int(ref.owner.tokens[ref.row])            # decode
                else:
                    last = req.all_token_ids[req.num_prefill_tokens - 1]  # (re)prefill done
                sampled[sr.request_id] = (last + 1) % self.vocab_size
        return {"sampled": sampled}

    @property
    def recoverable(self) -> bool:
        return True
