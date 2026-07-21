"""NpuModelRunner: SchedulerOutput -> batched TND NPU tensors -> paged FIA forward.

One FIA call over the whole mixed prefill+decode batch via block_table +
cumulative actual_seq_lengths; sparse_mode=3 (bottom-right causal) is correct for
prefill, chunked-prefill, and decode (q-len 1) alike. KV written to the paged
cache via _npu_reshape_and_cache by slot_mapping.

`_build` returns a `ForwardContext` consumed by the model's `forward(ctx)`. The
attention backend is resolved through the central registry, so this runner
contains no per-architecture branches.
"""
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from auto_infer.engine.executor import RunnerExecutor
from auto_infer.engine.execution import (
    BatchPlan, DeviceTokenBatch, ExecutionResult, ExecutionStats)
from auto_infer.engine.token_layout import slot_mapping
from auto_infer.worker.async_output import PinnedTokenBufferPool, enqueue_host_copy
from auto_infer.worker.staging import splice_device_tokens
from auto_infer.forward_context import ForwardContext
from auto_infer.worker.mtp_runner import MtpDrafter, MtpItem


class NpuModelRunner:
    def __init__(self, model, num_blocks: int, block_size: int,
                 max_num_batched_tokens: int = 8192, max_num_seqs: int = 256,
                 max_model_len: int = 4096, attention=None,
                 num_speculative_tokens: int = 1):
        self.model = model
        self.device = model.device
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.mtp = None                    # lazily-built MtpDrafter (spec-decode MTP proposer)
        self.num_speculative_tokens = num_speculative_tokens
        if attention is None:
            from auto_infer.layers.attention.registry import build_attention_backend
            attention = build_attention_backend(model, "paged", num_blocks, block_size)
        self.attn_backend, self.kv_caches = attention
        # Causal template mask for sparse_mode=3. MUST be 2048x2048: the CANN FIA op
        # requires maskDim==2048 for sparse_mode=3 (a COMPRESSED bottom-right causal
        # template, NOT per-position). Longer sequences are handled by sparse_mode +
        # actual_seq_lengths, so this is a hard op contract, not a length cap.
        self._mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device=self.device), diagonal=1)

        # Persistent input buffers (device) + pinned numpy host staging. Every step
        # fills a numpy staging array vectorized, then does ONE H2D `copy_` per buffer
        # into its used prefix. `_build` returns SLICES of these buffers (views, not
        # copies) so `submit`'s decode-splice write (`tok[fidx] = prev_sampled[rid]`)
        # mutates the persistent buffer directly. The ctor caps are only the INITIAL
        # size: `_ensure_capacity` auto-grows the buffers to any step's high-water
        # mark, so there is no fixed-cap overflow (steady workloads stop reallocating
        # after warmup). Graph capture must use its own stable per-gear buffers —
        # these eager buffers may move on growth.
        self._buf_tokens = self._buf_seqs = self._buf_blocks = 0
        self._alloc_buffers(max_num_batched_tokens, max_num_seqs,
                            (max_model_len + block_size - 1) // block_size)
        # Async output resources are lazy: D2H is queued onto an explicit copy
        # stream and the worker waits only on its ready event.
        self._output_thread: ThreadPoolExecutor | None = None
        self._copy_stream = None
        self._copy_pool = PinnedTokenBufferPool(pin_memory=self.device.type != "cpu")

    def close(self) -> None:
        if self._output_thread is not None:
            self._output_thread.shutdown(wait=True)
            self._output_thread = None

    def _make_staging(self, shape, torch_dtype):
        t = torch.empty(shape, dtype=torch_dtype)
        if self.device.type != "cpu":                 # pin only when a real H2D happens
            try:
                t = t.pin_memory()
            except (RuntimeError, NotImplementedError):
                pass
        return t, t.numpy()

    def _alloc_buffers(self, n_tokens: int, n_seqs: int, n_blocks: int) -> None:
        """(Re)allocate persistent input buffers + pinned staging to at least the
        given sizes (grow-only: takes max with current). Called from __init__ with
        the configured caps and from `_ensure_capacity` when a step exceeds them."""
        n_tokens = max(n_tokens, self._buf_tokens)
        n_seqs = max(n_seqs, self._buf_seqs)
        n_blocks = max(n_blocks, self._buf_blocks)
        dev = self.device
        self.token_ids_buf = torch.zeros(n_tokens, dtype=torch.int64, device=dev)
        self.positions_buf = torch.zeros(n_tokens, dtype=torch.int64, device=dev)
        self.slot_mapping_buf = torch.zeros(n_tokens, dtype=torch.int32, device=dev)
        self.block_table_buf = torch.zeros((n_seqs, n_blocks), dtype=torch.int32, device=dev)
        self._tok_stage, self._tok_stage_np = self._make_staging((n_tokens,), torch.int64)
        self._pos_stage, self._pos_stage_np = self._make_staging((n_tokens,), torch.int64)
        self._slot_stage, self._slot_stage_np = self._make_staging((n_tokens,), torch.int32)
        self._bt_stage, self._bt_stage_np = self._make_staging((n_seqs, n_blocks), torch.int32)
        self._buf_tokens, self._buf_seqs, self._buf_blocks = n_tokens, n_seqs, n_blocks

    def _ensure_capacity(self, T: int, num_reqs: int, max_blocks_step: int) -> None:
        """Grow the persistent buffers (2x the exceeded dim, to amortize) if this
        step is larger than the current capacity. No-op in steady state."""
        if T <= self._buf_tokens and num_reqs <= self._buf_seqs \
                and max_blocks_step <= self._buf_blocks:
            return
        nt = max(self._buf_tokens, T * 2) if T > self._buf_tokens else self._buf_tokens
        ns = max(self._buf_seqs, num_reqs * 2) if num_reqs > self._buf_seqs else self._buf_seqs
        nb = max(self._buf_blocks, max_blocks_step * 2) \
            if max_blocks_step > self._buf_blocks else self._buf_blocks
        self._alloc_buffers(nt, ns, nb)

    def _build(self, plan: BatchPlan):
        """Vectorized (numpy) fill of the persistent staging buffers + a single
        H2D `copy_` per buffer, then return a ForwardContext whose fields are
        SLICES (views) of the persistent device buffers. Per-request bookkeeping
        loops over `sched_output.scheduled` (bounded by batch size), but each
        request's token_ids/positions/slot_mapping are filled by one numpy slice
        assignment / vectorized gather regardless of token count."""
        bs = self.block_size
        cu_q, kv_lens = [], []
        sample_idx: dict[str, int] = {}
        decode_splice: list = []          # (flat_index, rid) for decode positions (device-fed token)
        reqs_meta = []                     # (rid, req, n, start, bt)
        qacc = 0
        max_blocks_step = 0
        all_decode = True
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            n = sr.num_tokens_to_compute
            start = req.num_computed_tokens
            bt = plan.block_tables[sr.request_id]
            reqs_meta.append((sr.request_id, req, n, start, bt, sr.is_prefill))
            qacc += n
            cu_q.append(qacc)                 # cumulative query lengths
            kv_lens.append(start + n)          # per-request kv length (NOT cumulative)
            sample_idx[sr.request_id] = qacc - 1
            max_blocks_step = max(max_blocks_step, len(bt))
        T = qacc
        num_reqs = len(reqs_meta)
        self._ensure_capacity(T, num_reqs, max_blocks_step)   # auto-grow; no fixed-cap overflow

        tok_np, pos_np, slot_np, bt_np = (
            self._tok_stage_np, self._pos_stage_np, self._slot_stage_np, self._bt_stage_np)
        bt_np[:num_reqs, :max_blocks_step] = 0            # block-table rows start from zeros

        cur = 0
        for r, (rid, req, n, start, bt, is_prefill) in enumerate(reqs_meta):
            end = start + n
            positions = np.arange(start, end, dtype=np.int64)
            if req.spec_draft and not is_prefill:   # spec DECODE: [last confirmed] + k drafts
                ids_arr = np.asarray(
                    [req.all_token_ids[start], *req.spec_draft], dtype=np.int64)
            else:
                ids_arr = np.asarray(req.all_token_ids[start:end], dtype=np.int64)
            bt_arr = np.asarray(bt, dtype=np.int64)
            slots = slot_mapping(bt_arr, positions, bs)

            tok_np[cur:cur + n] = ids_arr
            pos_np[cur:cur + n] = positions
            slot_np[cur:cur + n] = slots
            bt_np[r, :len(bt)] = bt_arr

            decode_mask = positions >= req.num_prompt_tokens   # decode: token is a prior sample
            if decode_mask.any():
                for li in np.nonzero(decode_mask)[0]:
                    decode_splice.append((cur + int(li), rid))
            if not decode_mask.all():
                all_decode = False
            cur += n

        self.token_ids_buf[:T].copy_(self._tok_stage[:T])
        self.positions_buf[:T].copy_(self._pos_stage[:T])
        self.slot_mapping_buf[:T].copy_(self._slot_stage[:T])
        self.block_table_buf[:num_reqs, :max_blocks_step].copy_(
            self._bt_stage[:num_reqs, :max_blocks_step])

        ctx = ForwardContext(
            token_ids=self.token_ids_buf[:T],
            positions=self.positions_buf[:T],
            slot_mapping=self.slot_mapping_buf[:T],
            block_table=self.block_table_buf[:num_reqs, :max_blocks_step],
            cu_seqlens_q=cu_q,
            seqlens_kv=kv_lens,
            attn_mask=self._mask,
            attn_backend=self.attn_backend,
            kv_caches=self.kv_caches,
            is_decode=all_decode,
        )
        return ctx, sample_idx, decode_splice

    def submit(self, plan: BatchPlan, prev_sampled=None):
        """Enqueue the forward (no host sync). Decode-position input tokens are spliced
        from prev_sampled (device scalars from the prior batch) so the launch needs no
        CPU token round-trip. Returns a handle; sampled_of/collect read it."""
        if not plan.scheduled:
            return None
        prev_sampled = prev_sampled or {}
        ctx, sample_idx, decode_splice = self._build(plan)
        splice_device_tokens(
            ctx.token_ids,
            [item[0] for item in decode_splice],
            [item[1] for item in decode_splice],
            prev_sampled)
        hidden = self.model.forward(ctx)
        logits = self.model.logits(hidden)
        # rows that produced a sample this step (prompt finished)
        rows, reqs = [], []
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            if req.num_computed_tokens + sr.num_tokens_to_compute >= req.num_prefill_tokens:
                rows.append(sample_idx[sr.request_id]); reqs.append(req)
        if not rows:
            return {"tokens": None, "order": []}
        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        sel = logits[torch.tensor(rows, device=logits.device)]
        t, order = build_sampling_tensors(reqs, logits.shape[-1], logits.device)
        tokens = sample_batched(sel, t)                       # (B,) device, no sync
        return {"tokens": tokens, "order": order}

    def sampled_of(self, handle) -> DeviceTokenBatch | None:
        if not handle or handle["tokens"] is None:
            return None
        return DeviceTokenBatch.from_output(handle["tokens"], handle["order"])

    def collect(self, handle) -> ExecutionResult:
        """Materialize sampled tokens on CPU (the host sync point)."""
        if not handle or handle["tokens"] is None:
            return ExecutionResult()
        vals = handle["tokens"].tolist()                      # single D2H sync
        return ExecutionResult.from_single_tokens(
            {rid: int(v) for rid, v in zip(handle["order"], vals)})

    def _get_output_thread(self) -> ThreadPoolExecutor:
        if self._output_thread is None:
            self._output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="NpuOutputThread")
        return self._output_thread

    def collect_async(self, handle) -> Future:
        """Queue an event-ordered D2H, then let the worker wait/map host values."""
        if not handle or handle["tokens"] is None:
            fut: Future = Future()
            fut.set_result(ExecutionResult())
            return fut
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()
        copy = enqueue_host_copy(
            handle["tokens"], handle["order"], self._copy_stream,
            self._copy_pool, protect_source=False)
        return self._get_output_thread().submit(copy.result)

    def collect_result(self, future: Future) -> ExecutionResult:
        return future.result()

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        return self.collect(self.submit(plan, {}))

    def execute_spec_mtp(self, plan: BatchPlan) -> ExecutionResult:
        """Greedy MTP spec-decode step (batched, KV-reuse). One target forward over
        the batch → pre-norm hidden + verify (drafts came from the MTP head last
        step). Then the MTP head (its OWN paged KV) advances one position per
        NEWLY-CONFIRMED query position (using THIS step's hidden + the next token)
        and drafts each request's next token. Returns
        ({rid: emitted}, {rid: [next_draft]}, stats). Chunked-prefill / prefix-cache
        safe: every prefill chunk MTP-prefills its own positions (emitting only on
        completion); a prefix-cached span is skipped by both caches (MTP KV empty
        there — drafts degrade over it, never wrong, since verify emits the target's
        argmax)."""
        from auto_infer.spec_decode.rejection_sampler import verify_and_accept
        if self.mtp is None:
            from auto_infer.spec_decode.geometry import MtpGeometry
            geometry = MtpGeometry.recurrent_from_weights(
                self.model.w, self.num_speculative_tokens)
            self.mtp = MtpDrafter.from_model(
                self.model, self.num_blocks, self.block_size, geometry)
        ctx, _, _ = self._build(plan)
        h_norm, h_pre = self.model.forward_with_prenorm(ctx)
        preds = self.model.logits(h_norm).argmax(-1)                       # (T,) greedy

        emitted: dict[str, list[int]] = {}
        next_drafts: dict[str, list[int]] = {}
        mtp_items = []                     # (rid, hidden(n,H), next_tokens, abs_positions, block_table)
        accepted = steps = 0
        accepted_per_position = [0] * self.num_speculative_tokens
        start = 0
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            n = sr.num_tokens_to_compute
            qs, qe = start, start + n
            start = qe
            bt = plan.block_tables[sr.request_id]
            base = req.num_computed_tokens                                 # abs pos of query[0]
            toks = req.all_token_ids
            positions = list(range(base, base + n))
            if sr.is_prefill or not req.spec_draft:
                # prefill CHUNK / first decode: MTP-prefill these positions (token
                # after each = next prompt token; the final prompt position uses the
                # generated token). Emit only when the prompt completes — an
                # intermediate chunk advances KV without producing a token. A
                # prefix-cached span is skipped by both caches (MTP KV empty there;
                # drafts degrade, never wrong).
                complete = base + n >= req.num_prefill_tokens
                if complete:
                    tok0 = int(preds[qe - 1])
                    emitted[sr.request_id] = [tok0]
                    nxt = toks[base + 1:base + n] + [tok0]
                else:
                    nxt = toks[base + 1:base + n + 1]                      # intermediate chunk
                mtp_items.append(MtpItem(
                    sr.request_id, h_pre[qs:qe], list(nxt), positions, bt,
                    generate_drafts=complete))
            else:                                                          # spec decode: verify draft(s)
                seg = preds[qs:qe]                                         # p0..pk
                m_t, _, _ = verify_and_accept(
                    torch.tensor([req.spec_draft], device=preds.device), seg.unsqueeze(0))
                m = int(m_t[0])
                emit = seg[:m + 1].tolist()
                emitted[sr.request_id] = emit
                accepted += m
                steps += 1
                for position in range(m):
                    accepted_per_position[position] += 1
                mtp_items.append(MtpItem(
                    sr.request_id, h_pre[qs:qs + m + 1], emit,
                    list(range(base, base + m + 1)), bt))
        if mtp_items:
            next_drafts.update(self.mtp.draft(
                mtp_items, self.num_speculative_tokens))
        return ExecutionResult(
            tokens={rid: tuple(tokens) for rid, tokens in emitted.items()},
            next_drafts={rid: tuple(tokens) for rid, tokens in next_drafts.items()},
            stats=ExecutionStats(
                accepted=accepted, steps=steps,
                accepted_per_position=tuple(accepted_per_position)))


class PagedNpuExecutor(RunnerExecutor):
    """Executor backed by paged FIA NpuModelRunner. Drop-in for EngineCore."""

    def __init__(self, model_path: str, num_blocks: int, block_size: int,
                 device_index: int = 0, dtype: str = "bfloat16",
                 max_num_batched_tokens: int = 8192, max_num_seqs: int = 256,
                 max_model_len: int = 4096, num_speculative_tokens: int = 1):
        from auto_infer.engine.factory import load_model
        model = load_model(model_path, device_index, dtype)
        super().__init__(NpuModelRunner(
            model, num_blocks, block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs, max_model_len=max_model_len,
            num_speculative_tokens=num_speculative_tokens))
