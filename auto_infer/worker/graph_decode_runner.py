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
  * per step: fill static buffers (token/pos/slot/block_table), graph_task_update
    every layer handle with the batch's per-seq actual_seq_kvlen, then g.replay().
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

    def __init__(self, model, num_blocks, block_size, max_gear=32, max_model_len=4096,
                 force_eager=False):
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
        from auto_infer.layers.attention.registry import build_attention_backend
        self.scratch_blocks = _scratch_blocks_for_gears(max_gear)
        self.backend, self.caches = build_attention_backend(
            model, "graph", num_blocks + self.scratch_blocks, block_size)
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        self.max_gear = max_gear
        # bool mask for FIA-v2 (True = masked); 2048 is the CANN FIA sparse_mode=3
        # contract (compressed causal template; long seqs ride actual_seq_lengths)
        self.mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))
        self.gears: dict[int, _Gear] = {}
        self.prefill_gears: dict[int, _PrefillGear] = {}
        self.failed_prefill_gears: set[int] = set()
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
        self._copy_stream = None
        self._copy_pool = PinnedTokenBufferPool(pin_memory=self.device.type != "cpu")
        if (not self.force_eager
                and getattr(self.backend, "supports_prefill_graph", False)):
            self._prewarm_prefill_gears()

    def close(self) -> None:
        if self._output_thread is not None:
            self._output_thread.shutdown(wait=True)
            self._output_thread = None

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
        logits = self.model.logits(_gather_sample_hidden(hidden, rows))
        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        t, order = build_sampling_tensors(reqs, logits.shape[-1], logits.device)
        toks = sample_batched(logits, t)                       # (B,) device, NO D2H here
        return {"tokens": toks, "order": order}

    # ---------- graph (decode-only) ----------
    def _get_gear(self, B):
        g = _select_gear(B, self.max_gear)
        if g is None:
            return None
        if g not in self.gears:
            self.gears[g] = self._capture(g)
        return self.gears[g]

    def _get_prefill_gear(self, query_gear):
        """Runtime lookup only: online requests never pay graph capture cost."""
        return self.prefill_gears.get(query_gear)

    def _prewarm_prefill_gears(self):
        for query_gear in _prefill_capture_sizes(self.max_gear):
            self.stats["prefill_graph_capture_attempts"] += 1
            try:
                self.prefill_gears[query_gear] = self._capture_prefill(query_gear)
            except Exception:
                self.failed_prefill_gears.add(query_gear)
                self.stats["prefill_graph_capture_failures"] += 1

    def _capture_prefill(self, query_gear):
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
        torch.argmax(gear.logits, dim=-1, out=gear.sampled)

        graph = torch.npu.NPUGraph()
        self.backend.begin_capture()
        try:
            with torch.npu.graph(graph):
                hidden = self.model.forward(ctx)
                selected = hidden.index_select(0, gear.sample_rows)
                self.model.logits(selected, out=gear.logits)
                torch.argmax(gear.logits, dim=-1, out=gear.sampled)
        finally:
            self.backend.end_capture()
        gear.reg = self.backend.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(self.backend, torch.npu.Stream())
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
        torch.argmax(gear.logits, dim=-1, out=gear.sampled)
        graph = torch.npu.NPUGraph()
        self.backend.begin_capture()
        with torch.npu.graph(graph):
            hidden = self.model.forward(ctx)
            self.model.logits(hidden, out=gear.logits)
            torch.argmax(gear.logits, dim=-1, out=gear.sampled)
        self.backend.end_capture()
        gear.reg = self.backend.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(self.backend, torch.npu.Stream())
        from auto_infer.worker.decode_input_stager import DecodeInputStager
        gear.stager = DecodeInputStager(
            tid=gear.tid, positions=gear.ppos, slots=gear.pslot,
            block_table=gear.bt, block_size=self.block_size,
            scratch0=self._scratch0,
            active_token_mask=gear.active_token_mask)
        return gear

    def _graph_submit(self, plan, gear, prev_sampled):
        """Fill gears + graph.replay(), returning a handle holding the on-device
        sampled tensor (NO `.tolist()` D2H)."""
        reqs = plan.scheduled; B = len(reqs)
        staged = gear.stager.stage(plan)
        kvlens, order = staged.kv_lengths, staged.order
        # async decode-splice (see _eager_submit) — device-to-device scalar
        # assignment from `prev_sampled`, no host sync.
        gear.stager.splice(prev_sampled, order)
        ctx = self._make_ctx(gear.tid, gear.ppos, gear.pslot, gear.bt,
                             gear.qlen_cum, kvlens, is_decode=True,
                             active_token_mask=gear.active_token_mask)
        self.backend.reg = gear.reg                         # swap this gear's handles in
        gear.pipeline.replay(gear.graph, ctx)
        logits = gear.logits[:B]
        from auto_infer.worker.decode_epilogue import is_capturable_greedy
        reqs_o = [plan.get_request(rid) for rid in order]
        if is_capturable_greedy(reqs_o):
            toks = gear.sampled[:B]
            self.stats["captured_greedy_steps"] += 1
            self.stats["graph_steps"] += 1
            return {"tokens": toks, "order": order, "reused_output": True}
        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        t, _ = build_sampling_tensors(reqs_o, logits.shape[-1], logits.device)
        toks = sample_batched(logits, t)                     # (B,) device, NO D2H here
        self.stats["external_sampler_steps"] += 1
        self.stats["graph_steps"] += 1
        return {"tokens": toks, "order": order}

    def _prefill_graph_submit(self, plan, gear, prev_sampled):
        staged = gear.stager.stage(plan)
        gear.stager.splice(
            prev_sampled, staged.splice_order, staged.splice_rows)
        ctx = self._make_ctx(
            gear.token_ids, gear.positions, gear.slots, staged.block_table,
            staged.cumulative_query_lengths, staged.kv_lengths,
            is_decode=False, active_token_mask=gear.active_token_mask)
        self.backend.reg = gear.reg
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
                "order": staged.sample_order,
                "reused_output": True}

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
    def submit(self, plan: BatchPlan, prev_sampled=None):
        if not plan.scheduled:
            return None
        prev_sampled = prev_sampled or {}
        all_decode = all(
            sr.num_tokens_to_compute == 1
            and plan.get_request(sr.request_id).num_computed_tokens
            >= plan.get_request(sr.request_id).num_prefill_tokens
            for sr in plan.scheduled)
        B = len(plan.scheduled)
        gear = self._get_gear(B) if (all_decode and not self.force_eager) else None
        if gear is not None:
            return self._graph_submit(plan, gear, prev_sampled)
        if not all_decode and not self.force_eager and getattr(
                self.backend, "supports_prefill_graph", False):
            key = _select_prefill_gear(
                sum(item.num_tokens_to_compute for item in plan.scheduled),
                self.max_gear)
            if key is not None:
                gear = self._get_prefill_gear(key)
                if gear is not None:
                    return self._prefill_graph_submit(plan, gear, prev_sampled)
            self.stats["prefill_graph_fallbacks"] += 1
        return self._eager_submit(plan, prev_sampled)

    def sampled_of(self, handle) -> DeviceTokenBatch | None:
        if not handle or handle["tokens"] is None:
            return None
        tokens = handle["tokens"]
        if handle.get("reused_output"):
            # One batch copy (not per-row clones) gives retained refs stable
            # ownership after this gear's next captured replay overwrites its
            # fixed sampled buffer.
            tokens = tokens.clone()
        return DeviceTokenBatch.from_output(tokens, handle["order"])

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
            self._copy_pool, protect_source=handle.get("reused_output", False))
        return self._get_output_thread().submit(copy.result)

    def collect_result(self, future: Future) -> ExecutionResult:
        return future.result()

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        return self.collect(self.submit(plan, {}))


class GraphPagedNpuExecutor(RunnerExecutor):
    def __init__(self, model_path, num_blocks, block_size, device_index=0,
                 dtype="bfloat16", max_gear=32, max_model_len=4096,
                 force_eager=False):
        from auto_infer.engine.factory import load_model
        model = load_model(model_path, device_index, dtype)
        super().__init__(GraphPagedRunner(
            model, num_blocks, block_size, max_gear=max_gear,
            max_model_len=max_model_len, force_eager=force_eager))
