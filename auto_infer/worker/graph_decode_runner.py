"""ACL-graph decode runner (graph mode).

A drop-in Executor for EngineCore that runs the SAME paged engine (Scheduler +
KVCacheManager + continuous batching) but executes decode-only batches on a
captured ACL graph instead of eager, eliminating per-step host dispatch.

Graph vs eager is just which `AttentionBackend` (and `capturing` flag) is
injected into `model.forward(ctx)`, not a reimplemented forward. Key ops the
graph backend uses:
  * KV cache in NZ layout; KV written with npu_scatter_pa_kv_cache.
  * attention via npu_fused_infer_attention_score_v2.out (graph-capturable .out
    variant — no workspace sync, unlike the plain FIA which breaks capture).
  * whole-model decode forward captured once per *gear* batch size into an
    NPUGraph; each layer's FIA wrapped in graph_task_group_begin/end -> handle.
  * per step: stage static buffers, submit graph replay, then dispatch each
    layer's graph-task update on a dedicated host worker/update stream. Captured
    external events delay only the dependent attention nodes.
  * TND rule: actual_seq_qlen is CUMULATIVE, actual_seq_kvlen is PER-SEQUENCE.

The runner is model-agnostic: it resolves its graph backend through the central
attention registry, so the SAME
runner/gear/capture machinery drives Qwen2 (`GraphGqaBackend`) and DeepSeek
(`GraphMlaBackend`, MLA + MoE). MoE runs INSIDE the captured `model.forward(ctx)`
like any other FFN — the padded gear token count keeps its ops' shapes static,
with the dynamic per-expert distribution riding in the `group_list` device
tensor the grouped-GEMM ops read at replay.

Prefill/mixed shapes up to ``max_gear`` total query tokens use a separate,
startup-prewarmed family of covering token gears on the SAME NZ cache. Larger
shapes fall back to eager FIA-v2;
decode-only batches with B <= max_gear use the decode graph family.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import os
import time

import torch

from auto_infer.engine.executor import RunnerExecutor
from auto_infer.engine.execution import BatchPlan, DeviceTokenBatch, ExecutionResult
from auto_infer.engine.token_layout import slot_mapping
from auto_infer.worker.async_output import PinnedTokenBufferPool, enqueue_host_copy
from auto_infer.worker.staging import splice_device_tokens
from auto_infer.forward_context import ForwardContext

GEARS = [1, 2, 4, 8, 16, 32, 64]


def _select_gear(B, max_gear):
    """Pick the smallest captured-graph batch size (gear) that covers a
    decode-only step of B live requests, or None if B exceeds every gear
    <= max_gear (caller falls back to eager). Pure Python — host-testable."""
    return next((x for x in GEARS if x >= B and x <= max_gear), None)


def _decode_capture_sizes(max_gear):
    """Static decode graph sizes derived from the shared gear policy."""
    return [gear for gear in GEARS if gear <= max_gear]


def _prefill_capture_sizes(max_gear):
    """vLLM-compatible flattened-token graph sizes, capped by ``max_gear``."""
    sizes = [1, 2, 4]
    sizes += list(range(8, min(max_gear + 1, 256), 8))
    if max_gear >= 256:
        sizes += list(range(256, max_gear + 1, 16))
    return sorted({size for size in sizes if size <= max_gear})


def _select_prefill_gear(query_tokens, max_gear):
    if query_tokens < 1:
        return None
    return next((size for size in _prefill_capture_sizes(max_gear)
                 if size >= query_tokens), None)


def _prefill_graph_limit(max_prefill_tokens, max_gear, tp_world_size):
    """Bound TP startup capture to the serving batch gears.

    Capturing hundreds of full-model prefill shapes is useful on one device,
    but multiplies startup cost across every tensor-parallel rank. Larger TP
    prefills keep the eager fallback while common online shapes stay graphed.
    """
    return (max_prefill_tokens if tp_world_size == 1
            else min(max_prefill_tokens, max_gear))


def _scratch_blocks_for_gears(max_gear):
    """Cover the largest decode gear and every accepted exact prefill shape."""
    return max_gear


def _gather_sample_hidden(hidden, rows):
    """Select prompt-completing rows before the vocabulary projection."""
    index = torch.tensor(rows, dtype=torch.long, device=hidden.device)
    return hidden.index_select(0, index)


def _marshal_prefill_batch(scheduled, scheduler, block_size):
    """Host-side marshaling for a prefill/mixed/oversized-decode eager step:
    flattens every scheduled request's newly-computed tokens into TND rows.
    Pure Python (lists only, no device/NPU ops) — host-testable."""
    flat_ids, flat_pos, slots, cu_q, kv_lens, bt_rows = [], [], [], [], [], []
    sample_idx, qacc, maxb = {}, 0, 0
    for sr in scheduled:
        req = scheduler.get_request(sr.request_id)
        n = sr.num_tokens_to_compute; start = req.num_computed_tokens
        bt = scheduler.block_tables[sr.request_id]
        for j in range(n):
            pos = start + j
            flat_ids.append(req.all_token_ids[pos]); flat_pos.append(pos)
            slots.append(slot_mapping(bt, pos, block_size))
        qacc += n; cu_q.append(qacc); kv_lens.append(start + n)
        sample_idx[sr.request_id] = qacc - 1
        bt_rows.append(list(bt)); maxb = max(maxb, len(bt))
    return flat_ids, flat_pos, slots, cu_q, kv_lens, bt_rows, sample_idx, maxb


class _Gear:
    """One captured graph + its static buffers + per-layer handles (from the
    shared graph backend's `reg`, stashed here) for batch=g."""
    def __init__(self, g, max_blocks, vocab, device, dtype):
        self.g = g
        self.tid = torch.zeros(g, dtype=torch.long, device=device)
        self.ppos = torch.zeros(g, dtype=torch.long, device=device)
        self.pslot = torch.zeros(g, dtype=torch.int32, device=device)
        self.bt = torch.zeros(g, max_blocks, dtype=torch.int32, device=device)
        self.active_token_mask = torch.zeros(
            g, dtype=torch.bool, device=device)
        self.logits = torch.zeros(g, vocab, dtype=dtype, device=device)
        self.sampled = torch.zeros(g, dtype=torch.long, device=device)
        self.qlen_cum = list(range(1, g + 1))
        self.reg = []
        self.graph = None
        self.pipeline = None
        self.stager = None


class _PrefillGear:
    """One flattened-token graph shared by every compatible sequence count."""

    def __init__(self, query_gear, max_blocks, vocab, device, dtype):
        self.query_gear = query_gear
        self.token_ids = torch.zeros(query_gear, dtype=torch.long, device=device)
        self.positions = torch.zeros(query_gear, dtype=torch.long, device=device)
        self.slots = torch.zeros(query_gear, dtype=torch.int32, device=device)
        self.block_table = torch.zeros(
            query_gear, max_blocks, dtype=torch.int32, device=device)
        self.sample_rows = torch.zeros(
            query_gear, dtype=torch.long, device=device)
        self.active_token_mask = torch.zeros(
            query_gear, dtype=torch.bool, device=device)
        self.logits = torch.zeros(
            query_gear, vocab, dtype=dtype, device=device)
        self.sampled = torch.zeros(
            query_gear, dtype=torch.long, device=device)
        self.reg = []
        self.graph = None
        self.pipeline = None
        self.stager = None


class GraphPagedRunner:
    """Executor backed by NZ-cache paged attention with ACL-graph decode."""

    def __init__(self, model, num_blocks, block_size, max_gear=32,
                 max_prefill_tokens=256, max_model_len=4096,
                 force_eager=False, async_slots=1, max_num_seqs=256):
        self.force_eager = force_eager
        self.model = model
        self.device = model.device
        self.block_size = block_size
        self.num_blocks = num_blocks
        # Single shared NZ-layout backend/cache: both graph capture/replay and the
        # eager fallback read the SAME KV cache. Resolved by the attention registry.
        # We allocate one block per row of the largest capturable gear on top
        # of num_blocks and use those as scratch during lazy capture/padding:
        # the paged KVCacheManager (in EngineCore) only ever hands out 0..num_blocks-1,
        # so scratch (num_blocks..) can never overlap a live request's KV — regardless
        # of the allocator's fill order (it pops the free list from the tail).
        from auto_infer.distributed.parallel_state import tp_size
        from auto_infer.layers.attention.registry import build_attention_backend
        self.prefill_graph_limit = _prefill_graph_limit(
            max_prefill_tokens, max_gear, tp_size())
        self.scratch_blocks = _scratch_blocks_for_gears(
            max(max_gear, self.prefill_graph_limit))
        self.backend, self.caches = build_attention_backend(
            model, "graph", num_blocks + self.scratch_blocks, block_size)
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        self.max_gear = max_gear
        self.max_prefill_tokens = max_prefill_tokens
        # bool mask for FIA-v2 (True = masked); 2048 is the CANN FIA sparse_mode=3
        # contract (compressed causal template; long seqs ride actual_seq_lengths)
        self.mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))
        from auto_infer.worker.async_slots import (
            DeviceTokenStore, ExecutionSlotPool)
        self._slot_pool = ExecutionSlotPool(async_slots)
        self._slot_gears: list[dict[int, _Gear]] = [
            {} for _ in range(async_slots)]
        self.gears = self._slot_gears[0]
        self._stage_streams = [
            torch.npu.Stream() for _ in range(async_slots)]
        self._stage_dependencies = [
            torch.npu.Event() for _ in range(async_slots)]
        self._stage_ready = [
            torch.npu.Event() for _ in range(async_slots)]
        self._token_store = DeviceTokenStore(
            max_num_seqs * (async_slots + 1), self.device)
        self._async_slots = async_slots
        self.prefill_gears: dict[int, _PrefillGear] = {}
        self.failed_prefill_gears: set[int] = set()
        self._prefill_prewarm_active = False
        self._scratch0 = num_blocks       # scratch = the extra top region, disjoint from live KV
        self.stats = {"graph_steps": 0, "prefill_graph_steps": 0,
                      "prefill_graph_fallbacks": 0, "eager_steps": 0,
                      "prefill_graph_capture_attempts": 0,
                      "prefill_graph_capture_failures": 0,
                      "prefill_graph_online_captures": 0,
                      "captured_greedy_steps": 0, "external_sampler_steps": 0}
        # Single-worker output thread for the D2H `.tolist()`, created lazily. Same
        # default-stream-ordering assumption as NpuModelRunner.collect_async.
        self._output_thread: ThreadPoolExecutor | None = None
        self._task_update_pool = ThreadPoolExecutor(
            max_workers=async_slots, thread_name_prefix="GraphTaskUpdate")
        self._copy_stream = None
        self._copy_pool = PinnedTokenBufferPool(pin_memory=self.device.type != "cpu")
        if not self.force_eager:
            self._prewarm_decode_gears()
        if (self._async_slots == 1 and not self.force_eager
                and getattr(self.backend, "supports_prefill_graph", False)):
            self._prewarm_prefill_gears()

    def close(self) -> None:
        self._task_update_pool.shutdown(wait=True)
        if self._output_thread is not None:
            self._output_thread.shutdown(wait=True)
            self._output_thread = None

    def supports_async(self) -> bool:
        return True

    def _make_ctx(self, tid, ppos, slot, bt, cu_q, kv_lens, is_decode,
                  active_token_mask=None):
        return ForwardContext(
            token_ids=tid, positions=ppos, slot_mapping=slot, block_table=bt,
            cu_seqlens_q=cu_q, seqlens_kv=kv_lens, attn_mask=self.mask,
            attn_backend=self.backend, kv_caches=self.caches,
            is_decode=is_decode, active_token_mask=active_token_mask)

    # ---------- eager (prefill / mixed / oversized-decode) ----------
    def _eager_submit(self, plan, prev_sampled):
        """Marshal/forward/sample, returning a handle holding the on-device sampled
        tensor (NO `.tolist()` D2H) — the D2H moves to collect/collect_async so the
        caller can overlap it."""
        bs = self.block_size
        flat_ids, flat_pos, slots, cu_q, kv_lens, bt_rows, sample_idx, maxb = \
            _marshal_prefill_batch(plan.scheduled, plan, bs)
        dev = self.device
        bt_t = torch.zeros((len(bt_rows), maxb), dtype=torch.int32, device=dev)
        for r, row in enumerate(bt_rows):
            bt_t[r, :len(row)] = torch.tensor(row, dtype=torch.int32, device=dev)
        tid = torch.tensor(flat_ids, dtype=torch.long, device=dev)
        ppos = torch.tensor(flat_pos, dtype=torch.long, device=dev)
        slot = torch.tensor(slots, dtype=torch.int32, device=dev)
        # async decode-splice (mirrors NpuModelRunner.submit): a request whose
        # prefill already finished feeds its PRIOR sampled token (device
        # tensor from `prev_sampled`, no host sync) as this row's input,
        # instead of whatever `_marshal_prefill_batch` read out of
        # `req.all_token_ids` (which may still be the optimistic placeholder
        # EngineCore._advance_optimistic appended, in async mode). Gated on
        # dict membership only: a preempted/recomputing request is popped
        # from `prev_sampled` by the engine at preemption time, so its
        # re-fed *real* generated tokens (already in all_token_ids) are left
        # untouched here.
        splice_device_tokens(
            tid,
            [sample_idx[sr.request_id] for sr in plan.scheduled],
            [sr.request_id for sr in plan.scheduled],
            prev_sampled)
        self.backend.capturing = False
        ctx = self._make_ctx(tid, ppos, slot, bt_t, cu_q, kv_lens, is_decode=False)
        hidden = self.model.forward(ctx)
        # rows that produced a sample this step (prompt finished)
        rows, reqs = [], []
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            if req.num_computed_tokens + sr.num_tokens_to_compute >= req.num_prefill_tokens:
                rows.append(sample_idx[sr.request_id]); reqs.append(req)
        self.stats["eager_steps"] += 1
        if not rows:
            return {"tokens": None, "order": []}
        selected = _gather_sample_hidden(hidden, rows)
        logits = self.model.logits(selected)
        from auto_infer.layers.sampler import stable_greedy
        from auto_infer.worker.decode_epilogue import is_capturable_greedy
        if is_capturable_greedy(reqs):
            self.stats["captured_greedy_steps"] += 1
            return {
                "tokens": stable_greedy(
                    selected, logits, self.model.w["lm_head.weight"]),
                "order": [request.request_id for request in reqs],
            }
        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        t, order = build_sampling_tensors(reqs, logits.shape[-1], logits.device)
        toks = sample_batched(logits, t)                       # (B,) device, NO D2H here
        return {"tokens": toks, "order": order}

    # ---------- graph (decode-only) ----------
    def _get_gear(self, B, slot_id=0):
        g = _select_gear(B, self.max_gear)
        if g is None:
            return None
        return self._slot_gears[slot_id].get(g)

    def _prewarm_decode_gears(self):
        from auto_infer.distributed.parallel_state import tp_barrier
        for gears in self._slot_gears:
            for size in _decode_capture_sizes(self.max_gear):
                tp_barrier()
                gears[size] = self._capture(size)
                tp_barrier()

    def _get_prefill_gear(self, query_gear):
        """Runtime lookup only: online requests never pay graph capture cost."""
        return self.prefill_gears.get(query_gear)

    def _prewarm_prefill_gears(self):
        from auto_infer.distributed.parallel_state import tp_barrier, tp_size
        self._prefill_prewarm_active = True
        try:
            for query_gear in _prefill_capture_sizes(self.prefill_graph_limit):
                self.stats["prefill_graph_capture_attempts"] += 1
                error = None
                tp_barrier()
                try:
                    self.prefill_gears[query_gear] = self._capture_prefill(query_gear)
                except Exception as capture_error:
                    error = capture_error
                finally:
                    tp_barrier()
                if error is not None:
                    self.failed_prefill_gears.add(query_gear)
                    self.stats["prefill_graph_capture_failures"] += 1
                    if tp_size() > 1:
                        raise RuntimeError(
                            f"prefill graph capture failed for {query_gear}"
                        ) from error
        finally:
            self._prefill_prewarm_active = False

    def _capture_prefill(self, query_gear):
        if not self._prefill_prewarm_active:
            self.stats["prefill_graph_online_captures"] += 1
        cfg = self.model.cfg
        gear = _PrefillGear(
            query_gear, self.max_blocks, cfg.vocab_size,
            self.device, self.model.dtype)
        # Capture at maximum sequence count (one token per sequence). Runtime
        # graph-task updates replace the TND metadata and pass block_table[:B].
        initial_q = list(range(1, query_gear + 1))
        initial_kv = [1] * query_gear
        sample_rows = list(range(query_gear))
        for row in range(query_gear):
            block = self._scratch0 + row
            gear.slots[row] = block * self.block_size
            gear.block_table[row, 0] = block
        gear.sample_rows.copy_(torch.tensor(
            sample_rows, dtype=torch.long, device=self.device))
        gear.active_token_mask.fill_(True)
        ctx = self._make_ctx(
            gear.token_ids, gear.positions, gear.slots,
            gear.block_table[:query_gear],
            initial_q, initial_kv, is_decode=False,
            active_token_mask=gear.active_token_mask)

        self.backend.capturing = False
        hidden = self.model.forward(ctx)
        selected = hidden.index_select(0, gear.sample_rows)
        self.model.logits(selected, out=gear.logits)
        from auto_infer.layers.sampler import stable_greedy
        stable_greedy(
            selected, gear.logits, self.model.w["lm_head.weight"],
            out=gear.sampled)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        self.backend.begin_capture()
        try:
            with torch.npu.graph(graph):
                hidden = self.model.forward(ctx)
                selected = hidden.index_select(0, gear.sample_rows)
                self.model.logits(selected, out=gear.logits)
                stable_greedy(
                    selected, gear.logits, self.model.w["lm_head.weight"],
                    out=gear.sampled)
        finally:
            self.backend.end_capture()
        gear.reg = self.backend.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(
            self.backend, torch.npu.Stream(),
            metadata_slots=max(2, self._async_slots),
            registrations=gear.reg)
        from auto_infer.worker.prefill_input_stager import PrefillInputStager
        gear.stager = PrefillInputStager(
            token_ids=gear.token_ids, positions=gear.positions,
            slots=gear.slots, block_table=gear.block_table,
            sample_rows=gear.sample_rows, block_size=self.block_size,
            scratch0=self._scratch0,
            active_token_mask=gear.active_token_mask)
        return gear

    def _capture(self, g):
        cfg = self.model.cfg
        gear = _Gear(g, self.max_blocks, cfg.vocab_size,
                     self.device, self.model.dtype)
        bs = self.block_size
        # warmup + capture writing only into scratch blocks (never read by live reqs)
        for r in range(g):
            blk = self._scratch0 + r
            gear.tid[r] = 0; gear.ppos[r] = 0
            gear.pslot[r] = blk * bs
            gear.bt[r, 0] = blk
        kv0 = [1] * g
        gear.active_token_mask.fill_(True)
        ctx = self._make_ctx(gear.tid, gear.ppos, gear.pslot, gear.bt,
                             gear.qlen_cum, kv0, is_decode=True,
                             active_token_mask=gear.active_token_mask)
        self.backend.capturing = False
        hidden = self.model.forward(ctx)          # warmup (not captured)
        self.model.logits(hidden, out=gear.logits)
        from auto_infer.layers.sampler import stable_greedy
        stable_greedy(
            hidden, gear.logits, self.model.w["lm_head.weight"],
            out=gear.sampled)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        self.backend.begin_capture()
        with torch.npu.graph(graph):
            hidden = self.model.forward(ctx)
            self.model.logits(hidden, out=gear.logits)
            stable_greedy(
                hidden, gear.logits, self.model.w["lm_head.weight"],
                out=gear.sampled)
        self.backend.end_capture()
        gear.reg = self.backend.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(
            self.backend, torch.npu.Stream(),
            metadata_slots=max(2, self._async_slots),
            registrations=gear.reg)
        from auto_infer.worker.decode_input_stager import DecodeInputStager
        gear.stager = DecodeInputStager(
            tid=gear.tid, positions=gear.ppos, slots=gear.pslot,
            block_table=gear.bt, block_size=self.block_size,
            scratch0=self._scratch0,
            active_token_mask=gear.active_token_mask)
        return gear

    def _prepare_graph(self, plan, gear, prev_sampled, slot):
        """Stage immutable slot state and prepare metadata for replay."""
        reqs = plan.scheduled; B = len(reqs)
        compute_stream = torch.npu.current_stream()
        stage_stream = self._stage_streams[slot.slot_id]
        dependency = self._stage_dependencies[slot.slot_id]
        ready = self._stage_ready[slot.slot_id]
        # Record after all previously submitted compute. Metadata H2D can run
        # immediately on the slot stream; only the sampled-token splice waits
        # for the preceding graph's device result.
        dependency.record(compute_stream)
        with torch.npu.stream(stage_stream):
            staged = gear.stager.stage(plan)
            splice = gear.stager.prepare_splice(
                prev_sampled, staged.order)
            dependency.wait(stage_stream)
            gear.stager.apply_splice(splice)
            ready.record(stage_stream)
        kvlens, order = staged.kv_lengths, staged.order
        ctx = self._make_ctx(gear.tid, gear.ppos, gear.pslot, gear.bt,
                             gear.qlen_cum, kvlens, is_decode=True,
                             active_token_mask=gear.active_token_mask)
        ticket = gear.pipeline.prepare(ctx)
        logits = gear.logits[:B]
        from auto_infer.worker.decode_epilogue import is_capturable_greedy
        reqs_o = [plan.get_request(rid) for rid in order]
        if is_capturable_greedy(reqs_o):
            toks = gear.sampled[:B]
            sampler_tensors = None
        else:
            from auto_infer.layers.sampling_meta import build_sampling_tensors
            sampler_tensors, _ = build_sampling_tensors(
                reqs_o, logits.shape[-1], logits.device)
            toks = None
        return {
            "kind": "graph", "slot": slot, "gear": gear, "ticket": ticket,
            "submission_id": slot.sequence,
            "input_ready": ready,
            "tokens": toks, "logits": logits, "sampler_tensors": sampler_tensors,
            "order": order}

    def _submit_graph(self, prepared):
        gear = prepared["gear"]
        prepared["input_ready"].wait(torch.npu.current_stream())
        gear.pipeline.submit(gear.graph, prepared["ticket"])
        prepared["task_update"] = self._task_update_pool.submit(
            self._update_graph_task, prepared["submission_id"],
            gear.pipeline, prepared["ticket"])
        toks = prepared["tokens"]
        if toks is None:
            from auto_infer.layers.sampler import sample_batched
            toks = sample_batched(
                prepared["logits"], prepared["sampler_tensors"])
            self.stats["external_sampler_steps"] += 1
        else:
            self.stats["captured_greedy_steps"] += 1
        self.stats["graph_steps"] += 1
        prepared["tokens"] = toks
        prepared["token_batch"] = DeviceTokenBatch.from_output(
            toks, prepared["order"])
        return prepared

    @staticmethod
    def _update_graph_task(sequence, pipeline, ticket):
        span = nullcontext()
        if os.getenv("AUTO_INFER_ASYNC_TRACE", "") == "1":
            span = torch.profiler.record_function(
                f"auto_infer.async/{sequence}/task_update")
        started = time.time_ns()
        try:
            with span:
                pipeline.update(ticket)
        finally:
            if os.getenv("AUTO_INFER_ASYNC_TRACE", "") == "1":
                from auto_infer.engine.async_timeline import (
                    record_async_interval)
                record_async_interval(
                    sequence, "task_update", started, time.time_ns())

    def _prefill_graph_submit(self, plan, gear, prev_sampled):
        staged = gear.stager.stage(plan)
        gear.stager.splice(
            prev_sampled, staged.splice_order, staged.splice_rows)
        ctx = self._make_ctx(
            gear.token_ids, gear.positions, gear.slots, staged.block_table,
            staged.cumulative_query_lengths, staged.kv_lengths,
            is_decode=False, active_token_mask=gear.active_token_mask)
        gear.pipeline.replay(gear.graph, ctx)
        self.stats["prefill_graph_steps"] += 1
        if not staged.sample_order:
            return {"tokens": None, "order": []}

        count = len(staged.sample_order)
        reqs = [plan.get_request(rid) for rid in staged.sample_order]
        from auto_infer.worker.decode_epilogue import is_capturable_greedy
        if is_capturable_greedy(reqs):
            self.stats["captured_greedy_steps"] += 1
            return {
                "tokens": gear.sampled[:count],
                "order": staged.sample_order}

        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        logits = gear.logits[:count]
        tensors, order = build_sampling_tensors(
            reqs, logits.shape[-1], logits.device)
        self.stats["external_sampler_steps"] += 1
        return {"tokens": sample_batched(logits, tensors), "order": order}

    @staticmethod
    def _collect_handle(handle) -> dict[str, int]:
        """Materialize sampled tokens on CPU (the host sync point) — shared by
        both the eager and graph submit paths, since both return the same
        {"tokens": (B,) device tensor | None, "order": [rid, ...]} handle shape."""
        if not handle or handle["tokens"] is None:
            return {}
        vals = handle["tokens"].tolist()                     # single D2H sync
        return {rid: int(v) for rid, v in zip(handle["order"], vals)}

    def _get_output_thread(self) -> ThreadPoolExecutor:
        if self._output_thread is None:
            self._output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="GraphOutputThread")
        return self._output_thread

    # ---------- Executor protocol (submit/sampled_of/collect*) ----------
    def prepare(self, plan: BatchPlan, prev_sampled=None):
        if not plan.scheduled:
            return None
        prev_sampled = prev_sampled or {}
        slot = self._slot_pool.acquire()
        try:
            return self._prepare_in_slot(plan, prev_sampled, slot)
        except Exception:
            self._slot_pool.release(slot)
            raise

    def _prepare_in_slot(self, plan, prev_sampled, slot):
        all_decode = all(
            sr.num_tokens_to_compute == 1
            and plan.get_request(sr.request_id).num_computed_tokens
            >= plan.get_request(sr.request_id).num_prefill_tokens
            for sr in plan.scheduled)
        B = len(plan.scheduled)
        gear = (self._get_gear(B, slot.slot_id)
                if (all_decode and not self.force_eager) else None)
        if gear is not None:
            return self._prepare_graph(plan, gear, prev_sampled, slot)
        # Prefill graphs have one fixed-buffer family. Async execution uses the
        # private-tensor eager path as a correctness barrier until prefill owns
        # the same replicated slot contract as decode.
        if (self._async_slots == 1 and not all_decode and not self.force_eager
                and getattr(
                    self.backend, "supports_prefill_graph", False)):
            key = _select_prefill_gear(
                sum(item.num_tokens_to_compute for item in plan.scheduled),
                self.prefill_graph_limit)
            if key is not None:
                gear = self._get_prefill_gear(key)
                if gear is not None:
                    handle = self._prefill_graph_submit(
                        plan, gear, prev_sampled)
                    handle.update(kind="ready", slot=slot)
                    return handle
            self.stats["prefill_graph_fallbacks"] += 1
        handle = self._eager_submit(plan, prev_sampled)
        handle.update(kind="ready", slot=slot)
        return handle

    def submit_prepared(self, prepared):
        if prepared is None:
            return None
        try:
            if prepared["kind"] == "graph":
                return self._submit_graph(prepared)
            tokens = prepared["tokens"]
            if tokens is not None:
                prepared["token_batch"] = DeviceTokenBatch.from_output(
                    tokens, prepared["order"])
            return prepared
        except Exception:
            self._slot_pool.release(prepared["slot"])
            prepared["slot"] = None
            raise

    def submit(self, plan: BatchPlan, prev_sampled=None):
        return self.submit_prepared(self.prepare(plan, prev_sampled))

    def sampled_of(self, handle) -> DeviceTokenBatch | None:
        if not handle or handle["tokens"] is None:
            return None
        return handle["token_batch"]

    def collect(self, handle) -> ExecutionResult:
        return ExecutionResult.from_single_tokens(self._collect_handle(handle))

    def collect_async(self, handle) -> Future:
        """Submit the D2H to the single-worker output thread. Relies on the same
        default-per-device-stream ordering assumption as
        NpuModelRunner.collect_async (see that docstring) — no explicit cross-stream
        event/wait here. `handle["tokens"]` is private to THIS step's
        `sample_batched` call (never written to by a later `submit`), so the output
        thread only ever reads it."""
        if not handle or handle["tokens"] is None:
            fut: Future = Future()
            fut.set_result(ExecutionResult())
            return fut
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()
        copy = enqueue_host_copy(
            handle["tokens"], handle["order"], self._copy_stream,
            self._copy_pool)
        return self._get_output_thread().submit(copy.result)

    def collect_result(self, future: Future) -> ExecutionResult:
        return future.result()

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        handle = self.submit(plan, {})
        try:
            return self.collect(handle)
        finally:
            self.release_submission(handle)

    def release_submission(self, handle) -> None:
        if handle is not None and handle.get("task_update") is not None:
            handle["task_update"].result()
        if handle is not None and handle.get("slot") is not None:
            self._slot_pool.release(handle["slot"])

    def stabilize_refs(self, handle, current_refs) -> dict:
        """Spill only requests skipped since this output slot was produced."""
        if handle is None or handle.get("token_batch") is None:
            return {}
        owner = handle["token_batch"]
        request_ids = []
        source_rows = []
        for rid, ref in current_refs.items():
            if ref.owner is owner:
                request_ids.append(rid)
                source_rows.append(ref.row)
        if not request_ids:
            return {}
        indices = torch.as_tensor(
            source_rows, dtype=torch.long, device=owner.tokens.device)
        tokens = owner.tokens.index_select(0, indices)
        return self._token_store.write(tokens, request_ids).refs()

    def release_requests(self, request_ids) -> None:
        self._token_store.release(request_ids)


class GraphPagedNpuExecutor(RunnerExecutor):
    def __init__(self, model_path, num_blocks, block_size, device_index=0,
                 dtype="bfloat16", max_gear=32, max_prefill_tokens=256,
                 max_model_len=4096, force_eager=False,
                 async_slots=1, max_num_seqs=256):
        from auto_infer.engine.factory import load_model
        model = load_model(model_path, device_index, dtype)
        super().__init__(GraphPagedRunner(
            model, num_blocks, block_size, max_gear=max_gear,
            max_prefill_tokens=max_prefill_tokens,
            max_model_len=max_model_len, force_eager=force_eager,
            async_slots=async_slots, max_num_seqs=max_num_seqs))
