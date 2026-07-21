import time
from collections import deque

from auto_infer.config import EngineConfig
from auto_infer.engine.executor import Executor
from auto_infer.engine.execution import BatchPlan
from auto_infer.errors import EngineStalledError, RequestRejectedError
from auto_infer.engine.kv_cache_manager import KVCacheManager
from auto_infer.engine.request import Request, RequestStatus
from auto_infer.engine.scheduler import Scheduler, SchedulerOutput
from auto_infer.engine.metrics import StatLogger


class EngineCore:
    def __init__(self, config: EngineConfig, executor: Executor):
        self.config = config
        self.kv = KVCacheManager(config.cache.num_blocks, config.cache.block_size)
        speculative_tokens = (
            config.spec_decode.num_speculative_tokens
            if config.spec_decode is not None else 0)
        self.scheduler = Scheduler(
            config.scheduler, self.kv, speculative_tokens)
        self.executor = executor
        # vLLM-aligned async pipeline state
        self._queue: deque = deque()               # in-flight (sched, handle)
        self._sampled = {}                         # rid -> last sampled token (device tensor/int)
        self._pending_idx: dict[str, deque] = {}   # rid -> placeholder output indices to backfill
        self.stat_logger = (StatLogger(config.log_stats_interval_s)
                            if config.log_stats else None)
        self.spec = config.spec_decode      # MTP spec-decode; drafts come from the runner's
        #                                     MTP head (per-req, stashed on req.spec_draft across steps)

    def add_request(self, req: Request) -> None:
        requested_length = req.num_prompt_tokens + req.sampling.max_tokens
        if requested_length > self.config.model.max_model_len:
            raise RequestRejectedError(
                f"request {req.request_id} has length {requested_length}; "
                f"max_model_len is {self.config.model.max_model_len}")
        if self.spec is not None and not self._supports_spec_sampling(req):
            raise RequestRejectedError(
                "MTP speculative decoding is greedy-only; sampling filters, "
                "penalties, and logit constraints are not supported")
        # The current allocator has no KV offload. Admit only requests whose
        # complete worst-case sequence can fit by itself; otherwise no amount
        # of preemption can make progress once the sole request reaches the
        # cache ceiling.
        continuation = (max(0, self.scheduler.num_speculative_tokens - 1)
                        if req.sampling.max_tokens > 1 else 0)
        max_kv_tokens = (req.num_prefill_tokens
                         + max(0, req.sampling.max_tokens - 1)
                         + continuation)
        required = self.kv.blocks_needed(max_kv_tokens)
        if required > self.kv.num_blocks:
            raise RequestRejectedError(
                f"request {req.request_id} requires {required} KV blocks; "
                f"capacity is {self.kv.num_blocks}")
        if self.stat_logger is not None and req.arrival_time is None:
            req.arrival_time = time.monotonic()
        self.scheduler.add_request(req)

    @staticmethod
    def _supports_spec_sampling(req: Request) -> bool:
        sampling = req.sampling
        return (
            sampling.temperature == 0
            and sampling.top_k == 0
            and sampling.top_p == 1
            and sampling.min_p == 0
            and sampling.presence_penalty == 0
            and sampling.frequency_penalty == 0
            and sampling.repetition_penalty == 1
            and sampling.logit_bias is None
            and sampling.bad_words_token_ids is None
            and sampling.allowed_token_ids is None
        )

    def _log_step(self, now, prefill_toks, gen_toks, finished) -> None:
        sl = self.stat_logger
        sl.record_step(prefill_toks, gen_toks)
        sl.record_finished(finished)
        sl.maybe_log(now, running=len(self.scheduler.running),
                     waiting=len(self.scheduler.waiting), kv=self.kv,
                     num_preemptions=self.scheduler.num_preemptions)

    def has_unfinished(self) -> bool:
        return bool(self._queue) or self.scheduler.has_unfinished()

    def finalized_output_count(self, request_id: str) -> int:
        """Length of the contiguous output prefix safe for external readers."""
        request = self.scheduler.get_request_or_none(request_id)
        if request is None:
            return 0
        pending = self._pending_idx.get(request_id)
        return pending[0] if pending else len(request.output_token_ids)

    def step(self) -> list[Request]:
        if self.spec is not None:
            return self._step_spec()
        if self.config.async_scheduling and self.executor.supports_async():
            return self._step_async()
        return self._step_sync()

    def _emit(self, req: Request, tokens: list[int], now: float) -> int:
        """Append emitted tokens (stop at max_tokens; is_finished catches EOS/stop
        mid-batch), set TTFT + promote on the first token. Returns #appended."""
        sl = self.stat_logger
        n = 0
        for t in tokens:
            if len(req.output_token_ids) >= req.sampling.max_tokens:
                break
            if sl and not req.output_token_ids and req.arrival_time is not None:
                req.first_token_time = now
                sl.record_ttft(now - req.arrival_time)
            req.append_output_token(t)
            n += 1
            self._promote(req)
            if req.is_finished():
                break
        return n

    # ---- MTP speculative decode step (greedy, KV-reuse; sync path) ----
    def _step_spec(self) -> list[Request]:
        sched = self._schedule_with_preemption()
        self._ensure_scheduled(sched)
        result = self.executor.execute_spec_mtp(BatchPlan.from_scheduler(sched, self.scheduler))
        self._validate_sync_result(sched, result)
        next_drafts, spec_stats = result.next_drafts, result.stats
        sl = self.stat_logger
        now = time.monotonic() if sl else 0.0
        prefill_toks = gen_toks = 0
        finished: list[Request] = []
        for sr in sched.scheduled:
            req = self.scheduler.get_request(sr.request_id)
            emitted = result.tokens.get(sr.request_id, ())
            if sr.is_prefill or not req.spec_draft:                 # prefill / no-draft: 1 token
                req.num_computed_tokens += sr.num_tokens_to_compute
                if sl and sr.is_prefill:
                    prefill_toks += sr.num_tokens_to_compute
                gen_toks += self._emit(req, emitted, now)
            else:                                                   # spec decode: m+1 tokens
                gen_toks += self._emit(req, emitted, now)
                req.num_computed_tokens = req.num_tokens - 1        # restore decode invariant
            if req.is_finished():
                finished.append(req)
        # Draft state is a one-step lease. Clear every scheduled row before
        # installing replacements so a backend omission cannot replay stale
        # proposals on the following step.
        for sr in sched.scheduled:
            r = self.scheduler.get_request_or_none(sr.request_id)
            if r is not None:
                r.spec_draft = []
        for rid, d in next_drafts.items():          # carry per-req MTP drafts to the next step
            r = self.scheduler.get_request_or_none(rid)
            if r is not None:
                r.spec_draft = list(d)
        self._release_finished(finished)
        if sl:
            sl.record_spec(
                spec_stats.steps, spec_stats.accepted,
                spec_stats.accepted_per_position)
            self._log_step(now, prefill_toks, gen_toks, len(finished))
        return finished

    def _safe_schedule(self) -> SchedulerOutput:
        """scheduler.schedule(), treating a kv.allocate()/append_slots() MemoryError
        as an implicit needs_preemption signal. The decode path defers gracefully
        (SchedulerOutput.needs_preemption); the prefill path's initial block
        allocation for a request (including a post-preempt_one() recompute) does
        not - it can raise directly, e.g. when a just-freed block is immediately
        claimed by a surviving running request's decode growth before the victim's
        own recompute-prefill runs on the very next schedule() call. Safe to catch
        and retry: KVCacheManager.allocate()/append_slots() check capacity before
        mutating any state, so a raised MemoryError leaves the scheduler/kv state
        exactly as it was (no partial block_tables entries to reconcile)."""
        try:
            return self.scheduler.schedule()
        except MemoryError:
            return SchedulerOutput(scheduled=[], num_batched_tokens=0, needs_preemption=True)

    def _discard_recomputed_state(self, request_id: str) -> None:
        request = self.scheduler.get_request_or_none(request_id)
        if request is not None:
            request.spec_draft = []
        self._sampled.pop(request_id, None)
        self._pending_idx.pop(request_id, None)

    def _schedule_with_preemption(self, defer_if_inflight: bool = False):
        sched = self._safe_schedule()
        while sched.needs_preemption:
            if defer_if_inflight and self._queue:
                break
            victim = self.scheduler.preempt_one()
            if victim is None:
                break
            self._discard_recomputed_state(victim)
            sched = self._safe_schedule()
        return sched

    def _release_finished(self, requests) -> None:
        for request in requests:
            request.status = RequestStatus.FINISHED
            self._pending_idx.pop(request.request_id, None)
            self._sampled.pop(request.request_id, None)
            self.scheduler.free_request(request.request_id)

    @staticmethod
    def _raise_result_errors(result) -> None:
        if result.errors:
            details = ", ".join(f"{rid}: {message}" for rid, message in result.errors.items())
            raise RuntimeError(f"executor failed requests: {details}")

    def _validate_sync_result(self, sched_output, result) -> None:
        self._raise_result_errors(result)
        for sr in sched_output.scheduled:
            req = self.scheduler.get_request(sr.request_id)
            produces_token = (req.num_computed_tokens + sr.num_tokens_to_compute >=
                              req.num_prefill_tokens)
            if produces_token and not result.tokens.get(sr.request_id):
                raise EngineStalledError(
                    f"executor did not return a token for request {sr.request_id}")

    def _ensure_scheduled(self, sched_output) -> None:
        if self.scheduler.has_unfinished() and not sched_output.scheduled:
            raise EngineStalledError("unfinished requests exist but no batch can be scheduled")

    # ---- synchronous step (schedule -> execute -> postprocess) ----
    def _step_sync(self) -> list[Request]:
        sched_output = self._schedule_with_preemption()
        self._ensure_scheduled(sched_output)
        result = self.executor.execute(BatchPlan.from_scheduler(sched_output, self.scheduler))
        self._validate_sync_result(sched_output, result)
        sampled = result.single_tokens()
        sl = self.stat_logger
        now = time.monotonic() if sl else 0.0
        prefill_toks = gen_toks = 0
        finished: list[Request] = []
        for sr in sched_output.scheduled:
            req = self.scheduler.get_request(sr.request_id)
            req.num_computed_tokens += sr.num_tokens_to_compute
            if sl and sr.is_prefill:
                prefill_toks += sr.num_tokens_to_compute
            if req.num_computed_tokens >= req.num_prefill_tokens and req.request_id in sampled:
                if sl:
                    gen_toks += 1
                    if not req.output_token_ids and req.arrival_time is not None:
                        req.first_token_time = now
                        sl.record_ttft(now - req.arrival_time)
                req.append_output_token(sampled[req.request_id])
                self._promote(req)
            if req.is_finished():
                finished.append(req)
        self._release_finished(finished)
        if sl:
            self._log_step(now, prefill_toks, gen_toks, len(finished))
        return finished

    # ---- vLLM-aligned async step (inter-step pipelining, output thread) ----
    # Fill a depth-`async_batches` queue with batches scheduled + submitted ahead
    # without blocking; state advances OPTIMISTICALLY after submit with a
    # placeholder token, and the real sampled token feeds the next batch's decode
    # input on-device (no CPU sync). The queue holds the output-thread FUTURE
    # (D2H runs on a dedicated output thread) so this thread builds the next batch
    # meanwhile; we block on the oldest future only when the queue is full, which
    # overlaps the queued batches' device compute and their already-issued host work.
    def _step_async(self) -> list[Request]:
        sl = self.stat_logger
        now = time.monotonic() if sl else 0.0
        prefill_toks = gen_toks = 0
        depth = max(1, self.config.async_batches)
        while len(self._queue) < depth and self._schedulable():
            sched = self._schedule_with_preemption(defer_if_inflight=True)
            if sched.needs_preemption:
                break
            if not sched.scheduled:
                break
            plan = BatchPlan.from_scheduler(sched, self.scheduler)
            handle = self.executor.submit(plan, self._sampled)
            # MERGE (not replace): a running decode request skipped this batch
            # (decode loop hit num_seqs >= max_num_seqs or budget < 1) keeps its
            # last-sampled token so the NEXT batch that does schedule it can
            # still find prev_sampled[rid]. Finished rids are pruned in
            # _finalize, so this can't grow unbounded.
            sampled_batch = self.executor.sampled_of(handle)
            if sampled_batch is not None:
                self._sampled.update(sampled_batch.refs())          # owners retain storage
            if sl:
                prefill_toks += sum(sr.num_tokens_to_compute
                                    for sr in sched.scheduled if sr.is_prefill)
            gen_toks += self._advance_optimistic(sched, now)
            # Hand the D2H off to the output thread and keep going — `handle`
            # is not touched again on this thread (its device tensor is only
            # ever read, never mutated, by the output thread's `.tolist()`).
            future = self.executor.collect_async(handle)
            self._queue.append((sched, future))
            if len(self._queue) < depth:
                continue
        if not self._queue:
            return []
        sched_old, future_old = self._queue.popleft()
        result = self.executor.collect_result(future_old)
        self._raise_result_errors(result)
        sampled = result.single_tokens()
        finished = self._finalize(sched_old, sampled)
        if sl:
            self._log_step(now, prefill_toks, gen_toks, len(finished))
        return finished

    def _schedulable(self) -> bool:
        for r in self.scheduler.waiting + self.scheduler.running:
            if len(r.output_token_ids) < r.sampling.max_tokens:
                return True
        return False

    def _advance_optimistic(self, sched, now=0.0) -> int:
        """Returns the number of requests that produced their (placeholder) first/
        next output token this batch — the step's generated-token count."""
        sl = self.stat_logger
        gen = 0
        for sr in sched.scheduled:
            req = self.scheduler.get_request(sr.request_id)
            prompt_done = req.num_computed_tokens + sr.num_tokens_to_compute >= req.num_prefill_tokens
            req.num_computed_tokens += sr.num_tokens_to_compute
            if prompt_done:
                if sl:
                    gen += 1
                    if not req.output_token_ids and req.arrival_time is not None:
                        req.first_token_time = now
                        sl.record_ttft(now - req.arrival_time)
                req.append_output_token(0)                     # placeholder, backfilled at finalize
                self._pending_idx.setdefault(req.request_id, deque()).append(len(req.output_token_ids) - 1)
                self._promote(req)
        return gen

    @staticmethod
    def _stops_at(req, tok, idx) -> bool:
        """Is `tok` (the real token just backfilled at output index `idx`) a
        terminal EOS/stop? Checked at the BACKFILL position, not output[-1] —
        async lookahead leaves placeholders after it, so [-1] would miss it."""
        sp = req.sampling
        if idx + 1 < sp.min_tokens:
            return False
        if not sp.ignore_eos and sp.eos_token_id is not None and tok == sp.eos_token_id:
            return True
        return tok in sp.stop_token_ids

    def _finalize(self, sched, sampled) -> list[Request]:
        finished: list[Request] = []
        for sr in sched.scheduled:
            rid = sr.request_id
            req = self.scheduler.get_request_or_none(rid)
            if req is None:                                     # already finished this drain
                continue
            if self._pending_idx.get(rid) and rid not in sampled:
                raise EngineStalledError(
                    f"executor did not return a token for request {rid}")
            if rid in sampled and self._pending_idx.get(rid):
                idx = self._pending_idx[rid].popleft()
                tok = sampled[rid]
                req.output_token_ids[idx] = tok                 # backfill real token
                if self._stops_at(req, tok, idx):               # EOS/stop lands mid-lookahead
                    del req.output_token_ids[idx + 1:]          # drop placeholders past the stop
                    self._pending_idx.pop(rid, None)
                    req.status = RequestStatus.FINISHED
                    finished.append(req)
                    continue
            # otherwise finish only once ALL in-flight backfills are applied
            # (else the last token could still be a placeholder).
            if req.is_finished() and not self._pending_idx.get(rid):
                req.status = RequestStatus.FINISHED
                finished.append(req)
        self._release_finished(finished)
        return finished

    def _promote(self, req: Request) -> None:
        self.scheduler.promote(req)

    def abort(self, request_id: str) -> None:
        """Cancel an in-flight request (e.g. client disconnected): free its KV and
        drop all engine-side state. Safe if the request is already finished or
        unknown. In-flight async batches that still reference it fall through the
        `get_request_or_none is None` guard in _finalize."""
        self._pending_idx.pop(request_id, None)
        self._sampled.pop(request_id, None)
        self.scheduler.free_request(request_id)
