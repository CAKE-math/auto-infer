"""Production model-derived MTP executor with prewarmed target and drafter graphs.

Drop-in for EngineCore with MTP speculative decode. A request-count target graph
verifies ``proposal_depth + 1`` rows per request and
compacts confirmed rows into fixed device buffers. A separately captured drafter
graph runs the trained MTP layer only over the covering confirmed-token gear.
Prefill and first decode are continuously batched across requests on the same NZ
caches. Both target and drafter project only sampling rows, while graph decode
keeps BF16 greedy heads inside capture. Dynamic FIA metadata uses an independent
update stream and two immutable metadata slots per graph gear.
"""
import torch

from auto_infer.engine.executor import Executor
from auto_infer.engine.execution import BatchPlan, ExecutionResult, ExecutionStats
from auto_infer.engine.token_layout import slot_mapping
from auto_infer.layers.mtp import RecurrentMtpHead
from auto_infer.layers.sampler import stable_greedy
from auto_infer.spec_decode.geometry import (
    MtpGeometry, validate_graph_mtp_depth)
from auto_infer.forward_context import ForwardContext
from auto_infer.worker.graph_decode_runner import GEARS     # one canonical gear ladder
from auto_infer.worker.graph_mtp_gears import (
    ContinuationGear as _ContinuationGear, DrafterGear as _DrafterGear,
    TargetGear as _TargetGear)
from auto_infer.worker.mtp_runner import MtpDrafter, MtpItem


def _decode_packed_results(rows, order, geometry):
    """Decode packed predictions, accepted count, and next drafts."""
    emitted, drafts = {}, {}
    accepted_total = 0
    for request_id, row in zip(order, rows):
        accepted = int(row[geometry.query_width])
        emitted[request_id] = list(row[:accepted + 1])
        draft_start = geometry.query_width + 1
        drafts[request_id] = [
            int(value)
            for value in row[draft_start:draft_start + geometry.draft_depth]
        ]
        accepted_total += accepted
    return emitted, drafts, accepted_total


def _next_mtp_tokens(tokens, base, count, complete, first_token):
    next_tokens = list(tokens[base + 1:base + count + (0 if complete else 1)])
    if complete:
        next_tokens.append(first_token)
    return next_tokens


def _finishes_after_one(request, token):
    """Predict Request.is_finished() after appending one token, without mutation."""
    params = request.sampling
    count = len(request.output_token_ids) + 1
    if count >= params.max_tokens:
        return True
    if count < params.min_tokens:
        return False
    if (not params.ignore_eos and params.eos_token_id is not None
            and token == params.eos_token_id):
        return True
    return token in params.stop_token_ids


def _chunk_spec_requests(requests, max_gear):
    largest = max(gear for gear in GEARS if gear <= max_gear)
    return [requests[start:start + largest]
            for start in range(0, len(requests), largest)]


def _select_drafter_gear(active_tokens, request_count, max_gear, geometry):
    request_gear = next(
        (gear for gear in GEARS
         if request_count <= gear <= max_gear), None)
    if request_gear is None:
        return None
    token_gear = next(
        (multiplier * request_gear
         for multiplier in range(1, geometry.query_width + 1)
         if active_tokens <= multiplier * request_gear), None)
    if token_gear is None:
        return None
    return token_gear, request_gear


def _reachable_drafter_pairs(max_gear, geometry):
    pairs = []
    for request_gear in (gear for gear in GEARS if gear <= max_gear):
        pairs.extend(
            (multiplier * request_gear, request_gear)
            for multiplier in range(1, geometry.query_width + 1)
        )
    return tuple(pairs)


class GraphMtpPagedRunner:
    def __init__(self, model, num_blocks, block_size, max_gear=16,
                 max_model_len=4096, num_speculative_tokens=1):
        self.model = model
        self.dev = model.device
        self.bs = block_size
        self.geometry = MtpGeometry.recurrent_from_weights(
            model.w, num_speculative_tokens)
        validate_graph_mtp_depth(self.geometry.draft_depth)
        if self.geometry.query_width > block_size:
            raise ValueError(
                "num_speculative_tokens must be smaller than block_size")
        self.MP = self.geometry.layer_prefix(0)
        cfg = model.cfg
        self.eps = cfg.rms_eps
        # Main + MTP KV reserve one scratch row per padded request plus enough
        # contiguous blocks for the compacted drafter tail.
        # region (num_blocks..), disjoint from the KVCacheManager's live 0..num_blocks-1.
        padding_blocks = (
            self.geometry.query_width * max_gear + block_size - 1
        ) // block_size
        nb_ext = num_blocks + max_gear + padding_blocks
        from auto_infer.layers.attention.registry import (
            build_attention_backend, build_mtp_attention_backend)
        self.main_be, self.main_kv = build_attention_backend(
            model, "graph", nb_ext, block_size)
        self.mtp_be, self.mtp_kv = build_mtp_attention_backend(
            model, "graph", self.MP, nb_ext, block_size)
        self.maxb = (max_model_len + self.geometry.draft_depth - 1
                     + block_size - 1) // block_size
        self.max_gear = max_gear
        self.mask = ~torch.tril(torch.ones((2048, 2048),   # CANN FIA sparse_mode=3 contract
                                           dtype=torch.bool, device=self.dev))
        self.mtp_head = RecurrentMtpHead(
            model, self.mtp_be, self.mtp_kv, self.mask, self.MP)
        self.mtp_drafter = MtpDrafter(
            self.mtp_head, device=self.dev, block_size=self.bs)
        self.target_gears: dict[int, _TargetGear] = {}
        self.drafter_gears: dict[tuple[int, int], _DrafterGear] = {}
        self.continuation_gears: dict[int, _ContinuationGear] = {}
        self._scratch0 = num_blocks       # scratch = extra top region, disjoint from live KV
        self._drafter_scratch0 = num_blocks + max_gear
        from auto_infer.worker.async_output import PinnedTokenBufferPool
        self._copy_stream = torch.npu.Stream()
        self._copy_pool = PinnedTokenBufferPool(
            pin_memory=self.dev.type != "cpu")
        self.stats = {
            "graph": 0, "eager": 0,
            "target_capture_attempts": 0,
            "drafter_capture_attempts": 0,
            "two_stage_steps": 0,
            "spec_steps": 0, "accepted": 0,
            "accepted_per_position": [0] * self.geometry.draft_depth,
        }
        self._prewarm_two_stage_gears()

    def _prewarm_two_stage_gears(self):
        for request_gear in (gear for gear in GEARS if gear <= self.max_gear):
            self.stats["target_capture_attempts"] += 1
            try:
                self.target_gears[request_gear] = self._capture_target(
                    request_gear)
            except Exception as exc:
                raise RuntimeError(
                    f"target graph capture failed for {request_gear}"
                ) from exc
        for key in _reachable_drafter_pairs(self.max_gear, self.geometry):
            self.stats["drafter_capture_attempts"] += 1
            try:
                self.drafter_gears[key] = self._capture_drafter(key)
            except Exception as exc:
                raise RuntimeError(
                    f"drafter graph capture failed for {key}"
                ) from exc
        if self.geometry.draft_depth > 1:
            for request_gear in (
                    gear for gear in GEARS if gear <= self.max_gear):
                self.continuation_gears[request_gear] = (
                    self._capture_continuation(request_gear))

    def _ctx(self, be, kv, tid, ppos, slot, bt, cu, kvlens,
             active_token_mask=None):
        return ForwardContext(token_ids=tid, positions=ppos, slot_mapping=slot, block_table=bt,
                              cu_seqlens_q=cu, seqlens_kv=kvlens, attn_mask=self.mask,
                              attn_backend=be, kv_caches=kv, is_decode=True,
                              active_token_mask=active_token_mask)

    def _copy_long_values(self, tensor):
        """Event-ordered D2H through a reusable pinned buffer."""
        shape = tuple(tensor.shape)
        count = tensor.numel()
        host = self._copy_pool.acquire(count)
        if tensor.device.type == "cpu":
            host[:count].copy_(tensor.reshape(-1))
        else:
            producer = torch.npu.current_stream()
            produced = torch.npu.Event()
            produced.record(producer)
            with torch.npu.stream(self._copy_stream):
                produced.wait(self._copy_stream)
                host[:count].copy_(tensor.reshape(-1), non_blocking=True)
                ready = torch.npu.Event()
                ready.record(self._copy_stream)
            ready.synchronize()
        values = host[:count].view(shape).tolist()
        self._copy_pool.release(host)
        return values

    def _mtp_hidden(self, hpre, next_tokens, ppos, slot, bt, cu, kvlens):
        """MTP head over its NZ KV (uses whatever capturing mode mtp_be is in):
        combine -> input_norm -> attention -> MLP, before final projection."""
        return self.mtp_head.hidden(
            hpre, next_tokens, ppos, slot, bt, cu, kvlens)

    def _target_body(self, gear, kvlens):
        """Verify target rows and compact only confirmed rows on device."""
        model = self.model
        width = gear.geometry.query_width
        ctx = self._ctx(
            self.main_be, self.main_kv, gear.tid, gear.ppos, gear.pslot,
            gear.bt, gear.cu, kvlens, gear.ep_active_token_mask)
        cos, sin = model._compute_cos_sin(gear.ppos)
        ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        hn, h_pre = model.forward_with_prenorm(ctx)
        logits = model.logits(hn)
        preds = stable_greedy(
            hn, logits, model.w["lm_head.weight"]).view(gear.g, width)
        accepted = torch.cumprod(
            (gear.drafts == preds[:, :-1]).to(torch.int32), 1).sum(1)
        accepted = accepted * gear.active_mask
        gear.p_buf.copy_(preds)
        gear.na_buf.copy_(accepted.to(torch.long))

        # Active requests form a prefix. Zero-length padding requests therefore
        # never perturb real owners, while the validity mask maps the fixed tail
        # to one dedicated scratch block.
        lengths = gear.active_mask * (1 + accepted)
        starts = lengths.cumsum(0) - lengths
        dest = torch.arange(gear.g * width, device=self.dev)
        owner = ((dest[:, None] >= starts[None, :]).sum(1) - 1).clamp_min(0)
        offset = dest - starts[owner]
        source = (width * owner + offset).clamp(0, gear.g * width - 1)
        valid = dest < lengths.sum()
        gear.compact_hidden.copy_(torch.where(
            valid[:, None], h_pre[source], torch.zeros_like(h_pre)))
        flat_preds = preds.reshape(-1)
        gear.compact_tokens.copy_(torch.where(
            valid, flat_preds[source], torch.zeros_like(dest)))
        gear.compact_positions.copy_(torch.where(
            valid, gear.ppos[source], gear.scratch_positions))
        gear.compact_slots.copy_(torch.where(
            valid, gear.pslot[source], gear.scratch_slots))

    def _drafter_body(self, gear, cu, kvlens, block_table):
        """Draft over the fixed confirmed-token gear and project only requests."""
        target = gear.target
        rows = gear.token_gear
        h = self._mtp_hidden(
            target.compact_hidden[:rows], target.compact_tokens[:rows],
            target.compact_positions[:rows], target.compact_slots[:rows],
            block_table, cu, kvlens)
        selected = h.index_select(0, gear.sample_rows)
        gear.state_buf.copy_(selected)
        logits = self.model.logits(selected)
        stable_greedy(
            selected, logits, self.model.w["lm_head.weight"],
            out=gear.draft_buf[:, 0])
        width = target.geometry.query_width
        target.result_buf[:, :width].copy_(target.p_buf)
        target.result_buf[:, width].copy_(target.na_buf)
        target.result_buf[:, width + 1:].copy_(gear.draft_buf)

    def _continuation_chain_body(self, gear, kv_by_step):
        """Capture every recurrent proposal in one graph replay."""
        hidden, token = gear.hidden_in, gear.token_in
        for index, kv_lengths in enumerate(kv_by_step):
            hidden = self._mtp_hidden(
                hidden, token, gear.positions[index], gear.slots[index],
                gear.block_table, gear.cu, kv_lengths)
            logits = self.model.logits(hidden)
            token = stable_greedy(
                hidden, logits, self.model.w["lm_head.weight"])
            gear.draft_out[:, index].copy_(token)

    # ---------------- two-stage startup capture ----------------
    def _capture_target(self, g):
        cfg = self.model.cfg
        gear = _TargetGear(
            g, self.maxb, cfg.hidden_size, self.dev, self.model.dtype,
            self.geometry)
        width = self.geometry.query_width
        gear.active_mask.fill_(1)
        gear.ep_active_token_mask.fill_(True)
        gear.scratch_slots.copy_(
            self._drafter_scratch0 * self.bs + gear.scratch_positions.to(
                torch.int32))
        for row in range(g):
            block = self._scratch0 + row
            start = row * width
            gear.bt[row, 0] = block
            gear.ppos[start:start + width] = torch.arange(width, device=self.dev)
            gear.pslot[start:start + width] = (
                block * self.bs
                + torch.arange(width, dtype=torch.int32, device=self.dev)
            )
        kv0 = [width] * g
        self.main_be.capturing = False
        self._target_body(gear, kv0)
        graph = torch.npu.NPUGraph()
        self.main_be.begin_capture()
        try:
            with torch.npu.graph(graph):
                self._target_body(gear, kv0)
        finally:
            self.main_be.end_capture()
        gear.reg = self.main_be.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(self.main_be, torch.npu.Stream())
        from auto_infer.worker.decode_input_stager import SpecDecodeInputStager
        gear.stager = SpecDecodeInputStager(
            tid=gear.tid, positions=gear.ppos, slots=gear.pslot,
            block_table=gear.bt, drafts=gear.drafts,
            active_mask=gear.active_mask, block_size=self.bs,
            scratch0=self._scratch0, geometry=self.geometry,
            active_token_mask=gear.ep_active_token_mask)
        return gear

    def _capture_drafter(self, key):
        token_gear, request_gear = key
        target = self.target_gears[request_gear]
        gear = _DrafterGear(
            key, target, self.maxb, self.dev)
        # Capture with an exact, valid scratch layout. Runtime graph-task
        # updates replace query/KV metadata and may add one padding sequence.
        qlen = token_gear // request_gear
        initial_cu = [qlen * (row + 1) for row in range(request_gear)]
        initial_kv = [qlen] * request_gear
        for row in range(request_gear):
            block = self._scratch0 + row
            gear.block_table[row, 0] = block
            start = row * qlen
            stop = start + qlen
            target.compact_positions[start:stop] = torch.arange(
                qlen, device=self.dev)
            target.compact_slots[start:stop] = torch.arange(
                block * self.bs, block * self.bs + qlen,
                dtype=torch.int32, device=self.dev)
            gear.sample_rows[row] = stop - 1
        ctx_bt = gear.block_table[:request_gear]
        self.mtp_be.capturing = False
        self._drafter_body(gear, initial_cu, initial_kv, ctx_bt)
        graph = torch.npu.NPUGraph()
        self.mtp_be.begin_capture()
        try:
            with torch.npu.graph(graph):
                self._drafter_body(gear, initial_cu, initial_kv, ctx_bt)
        finally:
            self.mtp_be.end_capture()
        gear.reg = self.mtp_be.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(self.mtp_be, torch.npu.Stream())
        from auto_infer.worker.mtp_pipeline_stager import MtpPipelineStager
        gear.stager = MtpPipelineStager(
            block_table=gear.block_table, sample_rows=gear.sample_rows,
            block_size=self.bs, scratch0=self._drafter_scratch0,
            token_capacity=token_gear, geometry=self.geometry)
        return gear

    def _capture_continuation(self, request_gear):
        gear = _ContinuationGear(
            request_gear, self.maxb, self.model.cfg.hidden_size,
            self.dev, self.model.dtype, self.geometry.draft_depth)
        steps = self.geometry.draft_depth - 1
        for row in range(request_gear):
            block = self._scratch0 + row
            gear.block_table[row, 0] = block
            for index in range(steps):
                step = index + 1
                gear.positions[index, row] = step
                gear.slots[index, row] = block * self.bs + step
        kv_by_step = [[step + 1] * request_gear
                      for step in range(1, steps + 1)]
        self.mtp_be.capturing = False
        self._continuation_chain_body(gear, kv_by_step)
        graph = torch.npu.NPUGraph()
        self.mtp_be.begin_capture()
        try:
            with torch.npu.graph(graph):
                self._continuation_chain_body(gear, kv_by_step)
        finally:
            self.mtp_be.end_capture()
        gear.reg = self.mtp_be.reg
        gear.graph = graph
        from auto_infer.worker.graph_task_pipeline import GraphTaskPipeline
        gear.pipeline = GraphTaskPipeline(self.mtp_be, torch.npu.Stream())
        from auto_infer.worker.decode_input_stager import ContinuationInputStager
        gear.stager = ContinuationInputStager(
            positions=gear.positions, slots=gear.slots,
            block_table=gear.block_table, block_size=self.bs,
            scratch0=self._scratch0)
        return gear

    def _run_continuations(self, drafter, reqs, plan, accepted):
        depth = self.geometry.draft_depth
        if depth <= 1:
            return
        request_gear = drafter.request_gear
        gear = self.continuation_gears[request_gear]
        gear.hidden_in.copy_(drafter.state_buf)
        gear.token_in.copy_(drafter.draft_buf[:, 0])
        kv_by_step = gear.stager.stage_all(
            plan, reqs, accepted=accepted)
        contexts = [
            self._ctx(
                self.mtp_be, self.mtp_kv, gear.token_in,
                gear.positions[index], gear.slots[index], gear.block_table,
                gear.cu, kv_lengths)
            for index, kv_lengths in enumerate(kv_by_step)
        ]
        self.mtp_be.reg = gear.reg
        gear.pipeline.replay_many(gear.graph, contexts)
        drafter.draft_buf[:, 1:depth].copy_(gear.draft_out)

    # ---------------- eager prefill / first-decode (+ MTP prefill) ----------------
    def _eager_prefill(self, reqs, plan):
        """Non-graph rows (prefill chunk / first decode): run the chunk eager on the
        NZ caches and MTP-prefill its positions. Emits a token only when the prompt
        completes (an intermediate chunk advances KV without emitting). Chunked-
        prefill / prefix-cache safe (see execute_spec_mtp). Returns emitted, drafts."""
        bs = self.bs
        emitted, next_drafts = {}, {}
        self.main_be.capturing = False
        self.mtp_be.capturing = False
        token_ids, positions, slots, block_rows = [], [], [], []
        cu, kv_lengths = [], []
        complete_rows = []
        total = 0
        for request_row, sr in enumerate(reqs):
            req = plan.get_request(sr.request_id)
            n = sr.num_tokens_to_compute
            base = req.num_computed_tokens
            toks = req.all_token_ids
            bt = plan.block_tables[sr.request_id]
            pos = list(range(base, base + n))
            token_ids.extend(toks[base:base + n])
            positions.extend(pos)
            slots.extend(slot_mapping(bt, p, bs) for p in pos)
            block_rows.append(list(bt) + [0] * (self.maxb - len(bt)))
            total += n
            cu.append(total)
            kv_lengths.append(base + n)
            complete = base + n >= req.num_prefill_tokens
            if complete:
                complete_rows.append((request_row, total - 1))

        tid = torch.tensor(token_ids, dtype=torch.long, device=self.dev)
        ppos = torch.tensor(positions, dtype=torch.long, device=self.dev)
        slot_t = torch.tensor(slots, dtype=torch.int32, device=self.dev)
        btt = torch.tensor(block_rows, dtype=torch.int32, device=self.dev)
        ctx = self._ctx(
            self.main_be, self.main_kv, tid, ppos, slot_t, btt, cu,
            kv_lengths)
        cos, sin = self.model._compute_cos_sin(ppos)
        ctx.cos, ctx.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        h_norm, h_pre = self.model.forward_with_prenorm(ctx)

        first_tokens = {}
        if complete_rows:
            sample_rows = torch.tensor(
                [row for _, row in complete_rows], dtype=torch.long,
                device=self.dev)
            selected = h_norm.index_select(0, sample_rows)
            logits = self.model.logits(selected)
            sampled = stable_greedy(
                selected, logits, self.model.w["lm_head.weight"]).tolist()
            first_tokens = {
                request_row: token
                for (request_row, _), token in zip(complete_rows, sampled)}

        for request_row, token in first_tokens.items():
            emitted[reqs[request_row].request_id] = [token]
        items = []
        for request_row, sr in enumerate(reqs):
            req = plan.get_request(sr.request_id)
            n = sr.num_tokens_to_compute
            base = req.num_computed_tokens
            complete = request_row in first_tokens
            first = first_tokens.get(request_row)
            next_tokens = _next_mtp_tokens(
                req.all_token_ids, base, n, complete, first)
            items.append(MtpItem(
                sr.request_id, h_pre[cu[request_row - 1] if request_row else 0:
                                     cu[request_row]], next_tokens,
                list(range(base, base + n)), plan.block_tables[sr.request_id],
                generate_drafts=complete and not _finishes_after_one(req, first)))
        next_drafts.update(self.mtp_drafter.draft(
            items, self.geometry.draft_depth))
        return emitted, next_drafts

    def _two_stage_graph_decode(self, reqs, plan, target):
        """Replay target, stage exact confirmed metadata, then replay drafter."""
        B = len(reqs)
        staged_target = target.stager.stage(plan, reqs)
        order = staged_target.order
        self.main_be.reg = target.reg
        target_ctx = self._ctx(
            self.main_be, self.main_kv, target.tid, target.ppos,
            target.pslot, target.bt, target.cu, staged_target.kv_lengths,
            target.ep_active_token_mask)
        target.pipeline.replay(target.graph, target_ctx)

        # This is the only inter-stage control transfer. It is deliberately a
        # tiny B-element copy; target hidden/tokens/positions/slots remain on
        # device in fixed-address compact buffers.
        accepted = self._copy_long_values(target.na_buf[:B])
        key = _select_drafter_gear(
            sum(accepted) + B, B, self.max_gear, self.geometry)
        drafter = self.drafter_gears.get(key) if key is not None else None
        if drafter is None:
            raise RuntimeError(f"no prewarmed MTP drafter for {key}")

        staged_drafter = drafter.stager.stage_drafter(
            plan, order, accepted, token_gear=drafter.token_gear,
            request_gear=drafter.request_gear)
        self.mtp_be.reg = drafter.reg
        drafter_ctx = self._ctx(
            self.mtp_be, self.mtp_kv,
            target.compact_tokens[:drafter.token_gear],
            target.compact_positions[:drafter.token_gear],
            target.compact_slots[:drafter.token_gear],
            staged_drafter.block_table,
            staged_drafter.cumulative_query_lengths,
            staged_drafter.kv_lengths)
        drafter.pipeline.replay(drafter.graph, drafter_ctx)
        self._run_continuations(drafter, reqs, plan, accepted)
        if self.geometry.draft_depth > 1:
            width = self.geometry.query_width
            target.result_buf[:, width + 1:].copy_(drafter.draft_buf)

        packed = self._copy_long_values(target.result_buf[:B])
        emitted, next_drafts, accepted_total = _decode_packed_results(
            packed, order, self.geometry)
        self.stats["two_stage_steps"] += 1
        accepted_per_position = tuple(
            sum(value > position for value in accepted)
            for position in range(self.geometry.draft_depth))
        return emitted, next_drafts, accepted_total, B, accepted_per_position

    # ---------------- EngineCore interface ----------------
    def execute_spec_mtp(self, plan: BatchPlan) -> ExecutionResult:
        prefill_reqs, decode_reqs = [], []
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            # prefill CHUNK (incl. not-yet-complete) or first decode -> eager (MTP-
            # prefills the chunk, emits only on completion); a request with a draft
            # is a spec-decode row -> graph.
            if sr.is_prefill or not req.spec_draft:
                prefill_reqs.append(sr)
            else:
                decode_reqs.append(sr)
        emitted, next_drafts = {}, {}
        accepted = steps = 0
        accepted_per_position = [0] * self.geometry.draft_depth
        if prefill_reqs:
            e, nd = self._eager_prefill(prefill_reqs, plan)
            emitted.update(e); next_drafts.update(nd)
            self.stats["eager"] += 1
        if decode_reqs:
            # Every row with a carried draft must execute verify/accept. Split
            # oversized/non-standard batches across the largest supported gear;
            # routing them through prefill would neither verify the draft nor
            # preserve the two-query-row shape contract.
            for chunk in _chunk_spec_requests(decode_reqs, self.max_gear):
                request_gear = next(
                    (gear for gear in GEARS
                     if len(chunk) <= gear <= self.max_gear), None)
                if request_gear is None:
                    raise RuntimeError(
                        "no prewarmed graph gear covers speculative decode")
                target = self.target_gears[request_gear]
                e, nd, chunk_acc, chunk_steps, chunk_by_position = self._two_stage_graph_decode(
                    chunk, plan, target)
                emitted.update(e); next_drafts.update(nd)
                accepted += chunk_acc
                steps += chunk_steps
                for position, count in enumerate(chunk_by_position):
                    accepted_per_position[position] += count
                self.stats["graph"] += 1
        self.stats["spec_steps"] += steps
        self.stats["accepted"] += accepted
        for position, count in enumerate(accepted_per_position):
            self.stats["accepted_per_position"][position] += count
        return ExecutionResult(
            tokens={rid: tuple(tokens) for rid, tokens in emitted.items()},
            next_drafts={rid: tuple(tokens) for rid, tokens in next_drafts.items()},
            stats=ExecutionStats(
                accepted=accepted, steps=steps,
                accepted_per_position=tuple(accepted_per_position)))


class GraphMtpPagedNpuExecutor(Executor):
    """Drop-in EngineCore executor: two-stage graph MTP speculative decode.
    Used with EngineCore + a SpecDecodeConfig; the engine calls execute_spec_mtp
    each step. supports_async=False (the spec path is synchronous)."""

    def __init__(self, model_path, num_blocks, block_size, device_index=0,
                 dtype="bfloat16", max_gear=16, max_model_len=4096,
                 num_speculative_tokens=1):
        from auto_infer.engine.factory import load_model
        model = load_model(model_path, device_index, dtype)
        self.runner = GraphMtpPagedRunner(
            model, num_blocks, block_size, max_gear=max_gear,
            max_model_len=max_model_len,
            num_speculative_tokens=num_speculative_tokens)

    def supports_async(self):
        return False

    def execute_spec_mtp(self, plan: BatchPlan) -> ExecutionResult:
        return self.runner.execute_spec_mtp(plan)
