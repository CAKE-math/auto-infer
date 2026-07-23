from dataclasses import dataclass

from auto_infer.config import SchedulerConfig
from auto_infer.engine.kv_cache_manager import KVCacheManager
from auto_infer.errors import RequestRejectedError
from auto_infer.engine.request import Request, RequestStatus


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    num_tokens_to_compute: int
    is_prefill: bool
    block_ids: list[int]


@dataclass(frozen=True)
class SchedulerOutput:
    scheduled: list[ScheduledRequest]
    num_batched_tokens: int
    needs_preemption: bool = False


class Scheduler:
    def __init__(self, config: SchedulerConfig, kv: KVCacheManager,
                 num_speculative_tokens: int = 0):
        self.config = config
        self.kv = kv
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.block_tables: dict[str, list[int]] = {}
        self._requests: dict[str, Request] = {}
        self.num_preemptions = 0        # cumulative, for StatLogger
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(self, req: Request) -> None:
        if req.request_id in self._requests:
            raise RequestRejectedError(f"duplicate request id: {req.request_id}")
        self.waiting.append(req)
        self._requests[req.request_id] = req

    def get_request(self, request_id: str) -> Request:
        return self._requests[request_id]

    def get_request_or_none(self, request_id: str) -> "Request | None":
        return self._requests.get(request_id)

    def promote(self, req: Request) -> None:
        """Move a request into the RUNNING set (from WAITING, or first admit).
        Idempotent; the engine calls this the step a request emits its first
        token instead of reaching into .waiting/.running directly."""
        req.status = RequestStatus.RUNNING
        if req in self.waiting:
            self.waiting.remove(req)
        if req not in self.running:
            self.running.append(req)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def retire_request(self, request_id: str) -> None:
        """Stop admitting a request while preserving its KV for in-flight work."""
        self.running = [r for r in self.running if r.request_id != request_id]
        self.waiting = [r for r in self.waiting if r.request_id != request_id]

    def reclaim_request(self, request_id: str) -> None:
        """Release a retired request after all submitted batches drop their lease."""
        req = self._requests.get(request_id)
        bt = self.block_tables.get(request_id, [])
        if req is not None and bt and self.config.enable_prefix_caching:
            self.kv.register_prefix(
                req.all_token_ids[:req.num_computed_tokens], bt)
        self.kv.free(self.block_tables.pop(request_id, []))
        self._requests.pop(request_id, None)
        self.running = [r for r in self.running if r.request_id != request_id]
        self.waiting = [r for r in self.waiting if r.request_id != request_id]

    def free_request(self, request_id: str) -> None:
        """Synchronous convenience for callers with no in-flight leases."""
        self.retire_request(request_id)
        self.reclaim_request(request_id)

    def schedule(self) -> SchedulerOutput:
        scheduled: list[ScheduledRequest] = []
        budget = self.config.max_num_batched_tokens
        num_seqs = 0

        # 1) decode: running requests, (1 + #drafts) query tokens each — 1 confirmed
        #    + k drafts verified in one forward. Per-request: n-gram always drafts k
        #    (padded); MTP drafts 1, or 0 on its first decode step (before the head
        #    has produced one), so a no-draft decode req is a plain 1-token step.
        needs_preempt = False
        for r in self.running:
            # A K-draft super-step reserves K+1 target rows and K-1 proposer
            # continuation rows. When <=K outputs remain, use a one-row step;
            # speculative work cannot be fully consumed and its transient KV
            # would exceed the admission bound.
            remaining_outputs = (
                r.sampling.max_tokens - len(r.output_token_ids))
            if (r.spec_draft
                    and remaining_outputs <= self.num_speculative_tokens):
                r.spec_draft = []
            qd = 1 + len(r.spec_draft)
            if num_seqs >= self.config.max_num_seqs or budget < qd:
                break
            if len(r.output_token_ids) >= r.sampling.max_tokens:
                continue                       # already scheduled up to max (async lookahead)
            bt = self.block_tables[r.request_id]
            continuation = max(0, len(r.spec_draft) - 1)
            need_blocks = self.kv.blocks_needed(
                r.num_tokens - 1 + qd + continuation)
            grows = need_blocks > len(bt)
            if grows and self.kv.num_free_blocks() < need_blocks - len(bt):
                needs_preempt = True
                break
            self.kv.append_slots(
                bt, r.num_tokens - 1, qd + continuation)
            scheduled.append(ScheduledRequest(r.request_id, qd, False, list(bt)))
            budget -= qd
            num_seqs += 1

        # 2) prefill: waiting requests, chunked, highest priority first (ties: arrival order)
        still_waiting: list[Request] = []
        # vLLM-compatible semantics: the long-prefill threshold caps each
        # request's chunk, not the aggregate batch. Treating it as a global
        # allowance serialized B requests into O(B) prefill scheduler rounds.
        per_request_prefill_cap = (
            (self.config.long_prefill_token_threshold or budget)
            if self.config.enable_chunked_prefill else budget)
        # priority desc, ties by arrival (waiting-list order). Sort (index, req)
        # pairs so the tie-break is O(1) per element, not a fresh .index() scan.
        waiting_order = [r for _, r in sorted(
            enumerate(self.waiting), key=lambda t: (-t[1].priority, t[0]))]
        for r in waiting_order:
            if num_seqs >= self.config.max_num_seqs or budget < 1:
                still_waiting.append(r)
                continue
            bt = self.block_tables.setdefault(r.request_id, [])
            if not bt:
                # cache at most num_prompt_tokens-1 tokens so >=1 token is always
                # computed to produce the first logits. Match runs
                # before the remaining/budget gate below so a non-chunked
                # request whose full prompt exceeds budget but whose uncached
                # remainder (after a prefix hit) fits isn't starved forever.
                #
                # No state (num_computed_tokens, bt) is mutated until the
                # remainder allocation is confirmed to succeed. Otherwise a
                # deferred request would keep its matched-prefix blocks in bt
                # forever (bt non-empty => `if not bt:` never re-enters), so it
                # would run later with a block table that only covers the
                # matched prefix - out-of-bounds block-table indexing / KV
                # corruption on the NPU decode path.
                matched = (self.kv.match_prefix(r.prompt_token_ids[:r.num_prompt_tokens - 1])
                           if self.config.enable_prefix_caching else [])
                have = len(matched) * self.kv.block_size
                # size against num_prefill_tokens (not num_prompt_tokens): for a
                # request recompute-preempted after generating output tokens,
                # num_prefill_tokens = num_tokens (prompt + generated so far) is
                # the true recompute target and exceeds num_prompt_tokens - under
                # -allocating here leaves the block table too small once decode
                # resumes, corrupting later register_prefix()/preempt_one() calls.
                # For a fresh request num_prefill_tokens == num_prompt_tokens, so
                # this is a no-op change for the non-preempted path.
                continuation = (max(0, self.num_speculative_tokens - 1)
                                if r.sampling.max_tokens > 1 else 0)
                need = r.num_prefill_tokens + continuation - have
                if need > 0 and not self.kv.can_allocate(need):
                    # Not enough free blocks for this request's initial (or
                    # post-preempt recompute) allocation - defer gracefully
                    # instead of letting kv.allocate() raise mid-loop (which
                    # would discard already-scheduled earlier requests in
                    # this same call, since the engine treats needs_preempt
                    # as an all-or-nothing signal and ignores `scheduled`).
                    # Roll back the prefix match: return the matched blocks to
                    # the evictable cache (free() undoes the refcount bump from
                    # match_prefix) and drop the empty block_tables entry so the
                    # request re-matches cleanly from scratch next time - it is
                    # left in a pristine (bt empty, num_computed_tokens
                    # unchanged) state.
                    self.kv.free(matched)
                    self.block_tables.pop(r.request_id, None)
                    still_waiting.append(r)
                    # Only flag needs_preemption when nothing else got
                    # scheduled this round either - otherwise this request
                    # simply waits for a later round (once already-scheduled
                    # work executes and either frees blocks or itself hits
                    # needs_preempt with an honestly-empty `scheduled`).
                    if not scheduled:
                        needs_preempt = True
                    continue
                if matched:
                    r.num_computed_tokens = have
                    bt.extend(matched)
                if need > 0:
                    bt.extend(self.kv.allocate(need))
                if self.config.enable_prefix_caching:   # prefix-hit rate (blocks), once per admit
                    self.kv.record_prefix_stats((r.num_prompt_tokens - 1) // self.kv.block_size,
                                                len(matched))
            remaining = r.num_prefill_tokens - r.num_computed_tokens
            if remaining <= 0:
                still_waiting.append(r)
                continue
            avail = min(budget, per_request_prefill_cap)
            if avail < 1:
                still_waiting.append(r)
                continue
            if self.config.enable_chunked_prefill:
                chunk = min(remaining, avail)
            else:
                if remaining > avail:
                    still_waiting.append(r)
                    continue
                chunk = remaining
            scheduled.append(ScheduledRequest(r.request_id, chunk, True, list(bt)))
            r.status = RequestStatus.RUNNING
            budget -= chunk
            num_seqs += 1
            still_waiting.append(r)  # stays until engine marks it running/decoding
        self.waiting = still_waiting

        used = self.config.max_num_batched_tokens - budget
        return SchedulerOutput(scheduled, used, needs_preemption=needs_preempt)

    def preempt_one(self) -> "str | None":
        """Recompute-style LIFO preemption: evict the most recently admitted
        running request, free its KV, and requeue it for full recompute.
        Returns the victim's request_id (so the engine can drop its now-invalid
        last-sampled token), or None if there is nothing to preempt."""
        if not self.running:
            return None
        self.num_preemptions += 1
        victim = self.running[-1]
        bt = self.block_tables.pop(victim.request_id, [])
        if bt and self.config.enable_prefix_caching:
            self.kv.register_prefix(
                victim.all_token_ids[:victim.num_computed_tokens], bt)
        self.kv.free(bt)
        victim.num_computed_tokens = 0
        victim.num_prefill_tokens = victim.num_tokens        # prompt + generated so far
        victim.status = RequestStatus.WAITING
        self.running.remove(victim)
        self.waiting.insert(0, victim)
        return victim.request_id
