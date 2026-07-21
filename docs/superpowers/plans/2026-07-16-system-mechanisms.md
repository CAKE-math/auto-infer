# System Mechanisms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up and complete the four "costly omissions" vs vLLM — prefix caching, batched sampling with a full sampling-math surface, scheduling policy, and preemption/recompute — then verify on real NPU (npu2).

**Architecture:** All changes cluster in `engine/` (scheduler, kv_cache_manager, engine_core, request), `layers/sampler.py`, and the two executors (`worker/model_runner.py`, `worker/graph_decode_runner.py`). Prefix caching becomes real via an evictable LRU block pool; sampling becomes a batched vectorized logits-processor chain; preemption is recompute-based and, in async mode, drain-then-preempt so it never touches an in-flight batch.

**Tech Stack:** Python 3.10+, PyTorch + torch_npu (Ascend CANN), pytest. Host-side logic (scheduler/kv/sampler-math) tested on CPU with MockExecutor; device wiring verified on npu2.

## Global Constraints

- Preemption KV handling: **recompute-based only** (no host KV swap).
- Preemption × async queue: **drain-then-preempt** — never free a block referenced by an in-flight batch; async fast-path unchanged when no memory pressure.
- SamplingParams scope: **sampling-math surface only** — no stop-strings, no logprobs, no n>1, no guided decoding.
- Sampling ops must stay mask/arithmetic-based (no host control flow inside the math) so they remain ACL-graph-compatible; sampling runs outside the captured graph as today.
- `num_free_blocks()` counts allocatable blocks = free + evictable-cached.
- Match at most `num_prompt_tokens - 1` prompt tokens as prefix, so ≥1 token is always computed to produce the first logits.
- Follow existing test style in `tests/` (module-level `def test_*`, `make_*`/`req` helpers, plain asserts).

---

## File Structure

- `engine/kv_cache_manager.py` — add evictable LRU pool; `free` retains registered blocks; `allocate` evicts LRU; `match_prefix` revives cached blocks. (Task 1)
- `engine/scheduler.py` — prefix match on admit; register on free; priority + prefill cap; `preempt_one` + `needs_preemption`. (Tasks 2, 6, 8)
- `engine/request.py` — expand `SamplingParams`; add `priority`, `num_prefill_tokens` to `Request`. (Tasks 3, 6, 7)
- `layers/sampler.py` — vectorized `SamplingTensors` + `sample_batched`. (Task 4)
- `engine/executor.py` — MockExecutor recompute-awareness. (Task 7)
- `worker/model_runner.py`, `worker/graph_decode_runner.py` — batched sampling wiring. (Task 5)
- `engine/engine_core.py` — `num_prefill_tokens` in prompt-done checks; drain-then-preempt. (Tasks 7, 9)
- `config/__init__.py` — `long_prefill_token_threshold`. (Task 6)
- `tests/` — unit tests per task. `scripts/` — two NPU verification scripts. (Task 10)

---

## Task 1: Evictable LRU block pool (spec §1a)

**Files:**
- Modify: `auto_infer/engine/kv_cache_manager.py`
- Test: `tests/test_kv_cache_manager.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `num_free_blocks() -> int` now returns `len(self._free) + len(self._cached)` (allocatable count).
  - `free(block_ids)` routes a registered block (has an entry in `_block_hash`) to an evictable LRU pool instead of the free list; unregistered blocks go straight to free.
  - `allocate`/`append_slots` evict the LRU cached block when `_free` is empty.
  - `match_prefix(token_ids)` revives a cached block on hit (moves it out of the pool, sets refcount to 1).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_kv_cache_manager.py`:

```python
from collections import OrderedDict  # noqa: F401  (only if needed by new asserts)


def test_free_retains_registered_block_for_reuse():
    m = KVCacheManager(num_blocks=10, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8]           # 2 full blocks
    blk = m.allocate(8)
    m.register_prefix(toks, blk)
    m.free(blk)                                # refcount 0 -> stays in evictable cache
    assert m.num_free_blocks() == 10           # counted as allocatable
    revived = m.match_prefix(toks)             # cache HIT after free (new behavior)
    assert revived == blk
    assert m.num_free_blocks() == 8            # revived blocks now held (ref=1)


def test_lru_eviction_unregisters_oldest():
    m = KVCacheManager(num_blocks=2, block_size=4)
    a = m.allocate(4)                          # block for prefix A
    m.register_prefix([1, 2, 3, 4], a)
    m.free(a)                                  # A cached (1 free real + 1 cached)
    # allocate 2 blocks: consumes the 1 real free + evicts cached A
    two = m.allocate(8)
    assert len(two) == 2
    assert m.num_free_blocks() == 0
    # A's hash was unregistered on eviction -> no longer matchable
    assert m.match_prefix([1, 2, 3, 4]) == []
```

Also UPDATE the last assertion of the existing `test_prefix_caching_share_and_release`:
change
```python
    assert m.match_prefix(toks) == []
```
to
```python
    assert m.match_prefix(toks) == blk         # freed blocks stay cached & revive
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_kv_cache_manager.py -q`
Expected: FAIL (new tests error / old assertion mismatch).

- [ ] **Step 3: Implement**

Rewrite `auto_infer/engine/kv_cache_manager.py` — replace `__init__`, `num_free_blocks`, `_alloc_one`, `allocate`, `append_slots`, `free`, `match_prefix` with:

```python
from collections import OrderedDict


class KVCacheManager:
    """Paged KV block allocator with prefix caching + an evictable LRU pool.

    A block is in exactly one state: free (unused), active (refcount >= 1), or
    cached (refcount 0 but still hash-registered, revivable by match_prefix, and
    evicted LRU-first only when free blocks run out)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._cached: "OrderedDict[int, int]" = OrderedDict()  # block_id -> hash, oldest first
        self._ref: dict[int, int] = {}
        self._hash_to_block: dict[int, int] = {}
        self._block_hash: dict[int, int] = {}

    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._cached)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks()

    def _evict_one(self) -> int:
        blk, hsh = self._cached.popitem(last=False)     # LRU: oldest freed
        if self._hash_to_block.get(hsh) == blk:
            del self._hash_to_block[hsh]
        self._block_hash.pop(blk, None)
        return blk

    def _alloc_one(self) -> int:
        blk = self._free.pop() if self._free else self._evict_one()
        self._ref[blk] = 1
        return blk

    def allocate(self, num_tokens: int) -> list[int]:
        n = self.blocks_needed(num_tokens)
        if n > self.num_free_blocks():
            raise MemoryError(f"need {n} blocks, {self.num_free_blocks()} free")
        return [self._alloc_one() for _ in range(n)]

    def append_slots(self, block_ids: list[int], cur_num_tokens: int,
                     num_new_tokens: int) -> list[int]:
        total_needed = self.blocks_needed(cur_num_tokens + num_new_tokens)
        extra = total_needed - len(block_ids)
        if extra <= 0:
            return []
        if extra > self.num_free_blocks():
            raise MemoryError(f"need {extra} more blocks, {self.num_free_blocks()} free")
        new = [self._alloc_one() for _ in range(extra)]
        block_ids.extend(new)
        return new

    def free(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if b not in self._ref:
                continue
            self._ref[b] -= 1
            if self._ref[b] <= 0:
                del self._ref[b]
                if b in self._block_hash:            # registered full block -> keep cached
                    self._cached[b] = self._block_hash[b]
                else:
                    self._free.append(b)

    @staticmethod
    def _block_content_hash(prev_hash: int, chunk: tuple[int, ...]) -> int:
        return hash((prev_hash, chunk))

    def match_prefix(self, token_ids: list[int]) -> list[int]:
        bs = self.block_size
        matched: list[int] = []
        prev = 0
        for start in range(0, (len(token_ids) // bs) * bs, bs):
            chunk = tuple(token_ids[start:start + bs])
            h = self._block_content_hash(prev, chunk)
            blk = self._hash_to_block.get(h)
            if blk is None:
                break
            if blk in self._cached:                  # revive freed-but-cached block
                del self._cached[blk]
                self._ref[blk] = 1
            elif blk in self._ref:                   # shared with a live request
                self._ref[blk] += 1
            else:
                break                                # stale mapping; stop matching
            matched.append(blk)
            prev = h
        return matched

    def register_prefix(self, token_ids: list[int], block_ids: list[int]) -> None:
        bs = self.block_size
        prev = 0
        for idx in range(len(token_ids) // bs):
            chunk = tuple(token_ids[idx * bs:(idx + 1) * bs])
            h = self._block_content_hash(prev, chunk)
            blk = block_ids[idx]
            self._hash_to_block.setdefault(h, blk)
            self._block_hash.setdefault(blk, h)
            prev = h
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_kv_cache_manager.py -q`
Expected: PASS (all, including updated assertion).

- [ ] **Step 5: Commit**

```bash
git add auto_infer/engine/kv_cache_manager.py tests/test_kv_cache_manager.py
git commit -m "feat(kv): evictable LRU block pool for real prefix reuse (spec §1a)"
```

---

## Task 2: Prefix match on admit + register on free (spec §1b, §1c)

**Files:**
- Modify: `auto_infer/engine/scheduler.py` (`schedule` prefill branch, `free_request`)
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `KVCacheManager.match_prefix`, `register_prefix`, `block_size` (Task 1).
- Produces: waiting requests get `num_computed_tokens` advanced to the cached-prefix boundary on first schedule; `free_request` registers a finished request's full blocks before freeing.

- [ ] **Step 1: Write failing test**

Add to `tests/test_scheduler.py`:

```python
def test_prefix_hit_skips_recompute():
    s = make_sched(max_tokens=64)                # block_size=4 in make_sched's KV
    # First request: prompt of 2 full blocks, run to finish so blocks register.
    r1 = req("a", 8, max_tokens=1)
    s.add_request(r1)
    s.schedule()                                 # prefill allocates 2 blocks for "a"
    r1.num_computed_tokens = 8
    r1.status = RequestStatus.RUNNING
    r1.append_output_token(99)
    s.running = [r1]; s.waiting = []
    s.free_request("a")                          # registers a's full blocks
    # Second request: identical prompt -> prefix should be matched (all but last token).
    r2 = req("b", 8, max_tokens=4)
    s.add_request(r2)
    out = s.schedule()
    sr = next(x for x in out.scheduled if x.request_id == "b")
    # 8 prompt tokens, match capped to 7 -> 1 full block (4 tokens) hit.
    assert r2.num_computed_tokens == 4
    assert sr.num_tokens_to_compute == 4         # only the uncached remainder
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_scheduler.py::test_prefix_hit_skips_recompute -v`
Expected: FAIL (`num_computed_tokens == 0`, computes 8).

- [ ] **Step 3: Implement**

In `auto_infer/engine/scheduler.py`, replace the prefill block-table allocation lines (currently):

```python
            bt = self.block_tables.setdefault(r.request_id, [])
            if not bt:
                bt.extend(self.kv.allocate(r.num_prompt_tokens))
```
with:

```python
            bt = self.block_tables.setdefault(r.request_id, [])
            if not bt:
                # cache at most num_prompt_tokens-1 tokens so >=1 token is always
                # computed to produce the first logits (spec §1b).
                matched = self.kv.match_prefix(r.prompt_token_ids[:r.num_prompt_tokens - 1])
                if matched:
                    r.num_computed_tokens = len(matched) * self.kv.block_size
                    bt.extend(matched)
                have = len(bt) * self.kv.block_size
                if have < r.num_prompt_tokens:
                    bt.extend(self.kv.allocate(r.num_prompt_tokens - have))
```

Then replace `free_request`:

```python
    def free_request(self, request_id: str) -> None:
        req = self._requests.get(request_id)
        bt = self.block_tables.get(request_id, [])
        if req is not None and bt:                       # register full blocks for reuse
            self.kv.register_prefix(req.all_token_ids, bt)
        self.kv.free(self.block_tables.pop(request_id, []))
        self._requests.pop(request_id, None)
        self.running = [r for r in self.running if r.request_id != request_id]
        self.waiting = [r for r in self.waiting if r.request_id != request_id]
```

Note: `remaining = r.num_prompt_tokens - r.num_computed_tokens` in the loop already computes only the uncached tail — no further change needed there.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_scheduler.py -q`
Expected: PASS (existing scheduler tests unaffected — they never register, so `match_prefix` returns []).

- [ ] **Step 5: Commit**

```bash
git add auto_infer/engine/scheduler.py tests/test_scheduler.py
git commit -m "feat(sched): wire prefix cache into schedule/free (spec §1b/§1c)"
```

---

## Task 3: Expand SamplingParams (spec §2a)

**Files:**
- Modify: `auto_infer/engine/request.py` (`SamplingParams`)
- Test: `tests/test_request.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SamplingParams` with fields `temperature, top_k, top_p, min_p, presence_penalty, frequency_penalty, repetition_penalty, logit_bias, bad_words_token_ids, allowed_token_ids, min_tokens, ignore_eos` (plus existing `max_tokens, stop_token_ids`). All optional with inert defaults (greedy, no penalties, no masks).

- [ ] **Step 1: Write failing test**

Add to `tests/test_request.py`:

```python
from auto_infer.engine.request import SamplingParams


def test_sampling_params_defaults_are_inert():
    p = SamplingParams()
    assert p.temperature == 0.0          # greedy
    assert p.top_k == 0 and p.top_p == 1.0 and p.min_p == 0.0
    assert p.presence_penalty == 0.0 and p.frequency_penalty == 0.0
    assert p.repetition_penalty == 1.0
    assert p.logit_bias is None and p.bad_words_token_ids is None
    assert p.allowed_token_ids is None
    assert p.min_tokens == 0 and p.ignore_eos is False


def test_sampling_params_accepts_full_surface():
    p = SamplingParams(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05,
                       presence_penalty=0.5, frequency_penalty=0.2,
                       repetition_penalty=1.1, logit_bias={5: -10.0},
                       bad_words_token_ids=[[7]], allowed_token_ids=[1, 2, 3],
                       min_tokens=4, ignore_eos=True)
    assert p.logit_bias[5] == -10.0 and p.allowed_token_ids == [1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_request.py -q`
Expected: FAIL (unexpected keyword arguments).

- [ ] **Step 3: Implement**

In `auto_infer/engine/request.py` replace the `SamplingParams` dataclass:

```python
@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: dict[int, float] | None = None
    bad_words_token_ids: list[list[int]] | None = None   # pre-tokenized (no str->tok here)
    allowed_token_ids: list[int] | None = None
    min_tokens: int = 0
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_request.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add auto_infer/engine/request.py tests/test_request.py
git commit -m "feat(sampling): expand SamplingParams to full sampling-math surface (spec §2a)"
```

---

## Task 4: Vectorized batched sampler (spec §2b)

**Files:**
- Modify: `auto_infer/layers/sampler.py`
- Test: `tests/test_sampler.py`

**Interfaces:**
- Consumes: nothing (pure torch, CPU-testable).
- Produces:
  - `@dataclass SamplingTensors` with per-row tensors: `temperature (B,)`, `top_k (B,) int`, `top_p (B,)`, `min_p (B,)`, `presence (B,)`, `frequency (B,)`, `repetition (B,)`, and optional `(B, vocab)` masks/counts: `occurrence_counts`, `prompt_presence`, `bias`, `disallowed_mask` (bool, True = forbid).
  - `sample_batched(logits: Tensor(B,vocab), t: SamplingTensors, generator=None) -> Tensor(B,)`.
- The existing scalar `greedy` and `sample` stay for back-compat.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sampler.py`:

```python
from auto_infer.layers.sampler import SamplingTensors, sample_batched


def _greedy_tensors(B, vocab):
    return SamplingTensors(
        temperature=torch.zeros(B), top_k=torch.zeros(B, dtype=torch.long),
        top_p=torch.ones(B), min_p=torch.zeros(B),
        presence=torch.zeros(B), frequency=torch.zeros(B), repetition=torch.ones(B),
        occurrence_counts=None, prompt_presence=None, bias=None, disallowed_mask=None)


def test_batched_greedy_matches_per_row():
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 0.0]])
    out = sample_batched(logits, _greedy_tensors(2, 3))
    assert out.tolist() == [1, 0]


def test_batched_disallowed_mask_forbids_token():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    t = _greedy_tensors(1, 3)
    t.disallowed_mask = torch.tensor([[True, False, False]])   # forbid the argmax
    assert sample_batched(logits, t).tolist() == [1]


def test_batched_repetition_penalty_demotes_seen_token():
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    t = _greedy_tensors(1, 3)
    t.repetition = torch.tensor([2.0])
    t.occurrence_counts = torch.tensor([[1.0, 0.0, 0.0]])       # token 0 already seen
    t.prompt_presence = torch.zeros(1, 3)
    # token 0 (positive logit) divided by 2 -> 1.0 < 1.9 -> token 1 wins
    assert sample_batched(logits, t).tolist() == [1]


def test_batched_logit_bias_added():
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    t = _greedy_tensors(1, 3)
    t.bias = torch.tensor([[0.0, 0.0, 5.0]])
    assert sample_batched(logits, t).tolist() == [2]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sampler.py -q`
Expected: FAIL (`ImportError: SamplingTensors`).

- [ ] **Step 3: Implement**

Append to `auto_infer/layers/sampler.py`:

```python
from dataclasses import dataclass


@dataclass
class SamplingTensors:
    temperature: torch.Tensor              # (B,)
    top_k: torch.Tensor                    # (B,) long, 0 = disabled
    top_p: torch.Tensor                    # (B,)
    min_p: torch.Tensor                    # (B,)
    presence: torch.Tensor                 # (B,)
    frequency: torch.Tensor                # (B,)
    repetition: torch.Tensor               # (B,)
    occurrence_counts: "torch.Tensor | None"   # (B, vocab) float counts of output tokens
    prompt_presence: "torch.Tensor | None"     # (B, vocab) 1.0 where token in prompt/output
    bias: "torch.Tensor | None"                # (B, vocab) additive
    disallowed_mask: "torch.Tensor | None"     # (B, vocab) bool, True = forbid


def _apply_penalties(logits, t):
    if t.occurrence_counts is not None:
        # repetition: positive logits / rep, negative * rep, only for seen tokens
        seen = (t.occurrence_counts > 0)
        if t.prompt_presence is not None:
            seen = seen | (t.prompt_presence > 0)
        rep = t.repetition.unsqueeze(-1)
        repd = torch.where(logits > 0, logits / rep, logits * rep)
        logits = torch.where(seen, repd, logits)
        logits = logits - t.frequency.unsqueeze(-1) * t.occurrence_counts
        presence_hit = seen.to(logits.dtype)
        logits = logits - t.presence.unsqueeze(-1) * presence_hit
    return logits


def sample_batched(logits: torch.Tensor, t: SamplingTensors,
                   generator: "torch.Generator | None" = None) -> torch.Tensor:
    """logits (B, vocab) -> token ids (B,). Fully vectorized, mask/arith only."""
    logits = logits.float()
    if t.bias is not None:
        logits = logits + t.bias
    logits = _apply_penalties(logits, t)
    if t.disallowed_mask is not None:
        logits = logits.masked_fill(t.disallowed_mask, float("-inf"))

    greedy_rows = t.temperature <= 0.0
    temp = torch.where(greedy_rows, torch.ones_like(t.temperature), t.temperature)
    scaled = logits / temp.unsqueeze(-1)

    # top_k per row (0 = disabled): mask everything below the row's k-th value
    if int(t.top_k.max()) > 0:
        vocab = scaled.shape[-1]
        kcap = torch.where(t.top_k > 0, t.top_k, torch.full_like(t.top_k, vocab))
        kcap = kcap.clamp(max=vocab)
        sorted_vals, _ = torch.sort(scaled, descending=True, dim=-1)
        kth = sorted_vals.gather(-1, (kcap - 1).clamp(min=0).unsqueeze(-1))
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

    if float(t.top_p.min()) < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) > t.top_p.unsqueeze(-1)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, sorted_idx, sorted_logits)

    if float(t.min_p.max()) > 0.0:
        probs = torch.softmax(scaled, dim=-1)
        top = probs.max(dim=-1, keepdim=True).values
        scaled = torch.where(probs < t.min_p.unsqueeze(-1) * top,
                             torch.full_like(scaled, float("-inf")), scaled)

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    greedy_tok = logits.argmax(dim=-1)
    return torch.where(greedy_rows, greedy_tok, sampled)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_sampler.py -q`
Expected: PASS (new + existing scalar tests).

- [ ] **Step 5: Commit**

```bash
git add auto_infer/layers/sampler.py tests/test_sampler.py
git commit -m "feat(sampling): vectorized batched logits-processor chain (spec §2b)"
```

---

## Task 5: Wire batched sampling into executors (spec §2c) — NPU-verified

**Files:**
- Create: `auto_infer/layers/sampling_meta.py` (host-side builder)
- Modify: `auto_infer/worker/model_runner.py`, `auto_infer/worker/graph_decode_runner.py`
- Test: `tests/test_sampling_meta.py`

**Interfaces:**
- Consumes: `SamplingTensors` (Task 4), `SamplingParams` (Task 3), `Request.output_token_ids`/`prompt_token_ids`.
- Produces: `build_sampling_tensors(reqs: list[Request], vocab: int, device) -> (SamplingTensors, order: list[str])` where `order[i]` is the request_id of row `i`. The two executors call it, run `sample_batched` once, keep the `(B,)` device tensor as `sampled_dev` and expose per-rid device views via `unbind` (no sync); `collect` materializes with a single `.tolist()`.

- [ ] **Step 1: Write failing test** (host-only part: the builder)

Create `tests/test_sampling_meta.py`:

```python
import torch
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.layers.sampling_meta import build_sampling_tensors


def _req(rid, temp=0.0, outs=(), bad=None):
    r = Request(request_id=rid, prompt_token_ids=[1, 2],
                sampling=SamplingParams(temperature=temp, bad_words_token_ids=bad))
    for tk in outs:
        r.append_output_token(tk)
    return r


def test_order_and_temperature_row_alignment():
    reqs = [_req("a", temp=0.0), _req("b", temp=0.7)]
    t, order = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert order == ["a", "b"]
    assert t.temperature.tolist() == [0.0, 0.7]


def test_occurrence_counts_from_outputs():
    reqs = [_req("a", outs=(3, 3, 5))]
    t, _ = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert t.occurrence_counts[0, 3].item() == 2.0
    assert t.occurrence_counts[0, 5].item() == 1.0


def test_bad_words_sets_disallowed_mask():
    reqs = [_req("a", bad=[[7]])]
    t, _ = build_sampling_tensors(reqs, vocab=8, device=torch.device("cpu"))
    assert bool(t.disallowed_mask[0, 7]) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sampling_meta.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the builder**

Create `auto_infer/layers/sampling_meta.py`:

```python
"""Host-side construction of batched SamplingTensors from scheduled requests."""
import torch

from auto_infer.layers.sampler import SamplingTensors


def build_sampling_tensors(reqs, vocab: int, device):
    """reqs: list of Request (one per sampled row, in output order).
    Returns (SamplingTensors on `device`, order: list[request_id])."""
    B = len(reqs)
    order = [r.request_id for r in reqs]
    sp = [r.sampling for r in reqs]

    def col(attr, default, dtype=torch.float32):
        return torch.tensor([getattr(p, attr, default) for p in sp], dtype=dtype, device=device)

    temperature = col("temperature", 0.0)
    top_k = col("top_k", 0, dtype=torch.long)
    top_p = col("top_p", 1.0)
    min_p = col("min_p", 0.0)
    presence = col("presence_penalty", 0.0)
    frequency = col("frequency_penalty", 0.0)
    repetition = col("repetition_penalty", 1.0)

    need_pen = any(p.presence_penalty or p.frequency_penalty or p.repetition_penalty != 1.0
                   for p in sp)
    occ = pres = None
    if need_pen:
        occ = torch.zeros(B, vocab, device=device)
        pres = torch.zeros(B, vocab, device=device)
        for i, r in enumerate(reqs):
            for tk in r.output_token_ids:
                occ[i, tk] += 1.0
            for tk in r.prompt_token_ids:
                pres[i, tk] = 1.0

    bias = None
    if any(p.logit_bias for p in sp):
        bias = torch.zeros(B, vocab, device=device)
        for i, p in enumerate(sp):
            for tk, b in (p.logit_bias or {}).items():
                bias[i, tk] += b

    disallowed = None
    for i, (p, r) in enumerate(zip(sp, reqs)):
        forbid = []
        if p.allowed_token_ids is not None:
            allowed = set(p.allowed_token_ids)
            forbid.extend(tk for tk in range(vocab) if tk not in allowed)
        for grp in (p.bad_words_token_ids or []):
            if len(grp) == 1:
                forbid.append(grp[0])
        # min_tokens / ignore_eos: block stop tokens until min_tokens reached
        if len(r.output_token_ids) < p.min_tokens or p.ignore_eos:
            forbid.extend(p.stop_token_ids)
        if forbid:
            if disallowed is None:
                disallowed = torch.zeros(B, vocab, dtype=torch.bool, device=device)
            disallowed[i, torch.tensor(sorted(set(forbid)), device=device)] = True

    return SamplingTensors(temperature, top_k, top_p, min_p, presence, frequency,
                           repetition, occ, pres, bias, disallowed), order
```

- [ ] **Step 4: Run to verify pass** (builder)

Run: `pytest tests/test_sampling_meta.py -q`
Expected: PASS.

- [ ] **Step 5: Wire into `model_runner.py`**

In `NpuModelRunner.submit`, replace the per-request `sampled_dev` argmax loop and the `collect`/`sampled_of` methods:

```python
    def submit(self, sched_output, scheduler, prev_sampled=None):
        if not sched_output.scheduled:
            return None
        prev_sampled = prev_sampled or {}
        inputs, sample_idx, decode_splice = self._build(sched_output, scheduler)
        tok = inputs["token_ids"]
        for fidx, rid in decode_splice:
            if rid in prev_sampled:
                tok[fidx] = prev_sampled[rid]
        logits = self.model.forward_paged(
            tok, inputs["positions"], inputs["slot_mapping"],
            inputs["block_table"], inputs["actual_seq_q"], inputs["actual_seq_kv"],
            self.kv_caches, self._mask)
        # rows that produced a sample this step (prompt finished)
        rows, reqs = [], []
        for sr in sched_output.scheduled:
            req = scheduler.get_request(sr.request_id)
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

    def sampled_of(self, handle) -> dict:
        if not handle or handle["tokens"] is None:
            return {}
        return {rid: tk for rid, tk in zip(handle["order"], handle["tokens"].unbind(0))}

    def collect(self, handle) -> dict[str, int]:
        if not handle or handle["tokens"] is None:
            return {}
        vals = handle["tokens"].tolist()                      # single D2H sync
        return {rid: int(v) for rid, v in zip(handle["order"], vals)}
```

Note: `req.num_prefill_tokens` is added in Task 7; until then it equals `num_prompt_tokens`. If implementing Task 5 before Task 7, temporarily use `req.num_prompt_tokens` and switch in Task 7.

- [ ] **Step 6: Wire into `graph_decode_runner.py`**

In `GraphPagedRunner._graph`, replace the tail (`logits = ... ; toks = logits.argmax(-1); return {...}`):

```python
        logits = self.adapter.logits(gear.hout[:B])
        from auto_infer.layers.sampling_meta import build_sampling_tensors
        from auto_infer.layers.sampler import sample_batched
        reqs = [scheduler.get_request(sr.request_id) for sr in sched_output.scheduled]
        t, _ = build_sampling_tensors(reqs, logits.shape[-1], logits.device)
        toks = sample_batched(logits, t)
        self.stats["graph_steps"] += 1
        return {rid: int(v) for rid, v in zip(order, toks.tolist())}   # one D2H
```

And in `GraphPagedRunner._eager`, replace the per-request `int(logits[...].argmax())` loop with the same `build_sampling_tensors` + `sample_batched` batched form over the finished rows (mirror the model_runner submit pattern).

- [ ] **Step 7: Verify on NPU (npu2)**

Run: `AI_DEVICE=0 python scripts/smoke_qwen2.py` and `python scripts/verify_qwen2_graphdecode_batched.py`
Expected: greedy outputs unchanged vs pre-change (temperature defaults to 0). Then a quick temperature run produces varied-but-valid tokens.

- [ ] **Step 8: Commit**

```bash
git add auto_infer/layers/sampling_meta.py tests/test_sampling_meta.py \
        auto_infer/worker/model_runner.py auto_infer/worker/graph_decode_runner.py
git commit -m "feat(sampling): batched on-device sampling + single-sync collect (spec §2c)"
```

---

## Task 6: Scheduling policy — priority + prefill cap (spec §3)

**Files:**
- Modify: `auto_infer/config/__init__.py` (`SchedulerConfig`), `auto_infer/engine/request.py` (`Request.priority`), `auto_infer/engine/scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SchedulerConfig.long_prefill_token_threshold: int = 0` (0 = disabled); `Request.priority: int = 0`; `schedule()` drains waiting in `(-priority, arrival)` order and caps total prefill tokens per step at the threshold when > 0.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_scheduler.py`:

```python
def test_priority_orders_waiting():
    s = make_sched(max_tokens=4, max_num_seqs=1)   # only one can prefill this step
    lo = req("lo", 4); lo.priority = 0
    hi = req("hi", 4); hi.priority = 5
    s.add_request(lo); s.add_request(hi)            # lo arrived first
    out = s.schedule()
    assert out.scheduled[0].request_id == "hi"     # higher priority wins


def test_long_prefill_token_cap():
    s = make_sched(max_tokens=64, long_prefill_token_threshold=8)
    s.add_request(req("a", 20))
    out = s.schedule()
    assert out.scheduled[0].num_tokens_to_compute == 8   # capped below budget
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_scheduler.py -k "priority or long_prefill" -v`
Expected: FAIL (`priority`/`long_prefill_token_threshold` unknown).

- [ ] **Step 3: Implement**

In `config/__init__.py`, add to `SchedulerConfig`:

```python
    long_prefill_token_threshold: int = 0   # per-step prefill token cap (0 = disabled)
```

In `request.py`, add to `Request` (after `num_computed_tokens`):

```python
    priority: int = 0
```

In `scheduler.py` `schedule()`:
- Before the prefill loop, order the waiting scan by priority:
```python
        waiting_order = sorted(self.waiting, key=lambda r: (-r.priority, self.waiting.index(r)))
```
  and iterate `for r in waiting_order:` instead of `for r in self.waiting:`. Keep rebuilding `still_waiting` as today (append the un-promoted requests); assign `self.waiting = [r for r in still_waiting]` preserving arrival order for the leftovers.
- Track a per-step prefill budget. At the top of the prefill section:
```python
        prefill_cap = self.config.long_prefill_token_threshold or budget
        prefill_used = 0
```
  and when computing the chunk, cap it:
```python
            avail = min(budget, prefill_cap - prefill_used)
            if avail < 1:
                still_waiting.append(r)
                continue
            if self.config.enable_chunked_prefill:
                chunk = min(remaining, avail)
            else:
                if remaining > avail:
                    still_waiting.append(r); continue
                chunk = remaining
            ...
            prefill_used += chunk
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_scheduler.py -q`
Expected: PASS (existing tests unaffected — priority default 0 preserves FCFS, threshold 0 disables cap).

- [ ] **Step 5: Commit**

```bash
git add auto_infer/config/__init__.py auto_infer/engine/request.py \
        auto_infer/engine/scheduler.py tests/test_scheduler.py
git commit -m "feat(sched): priority ordering + per-step prefill cap (spec §3)"
```

---

## Task 7: Recompute semantics — `num_prefill_tokens` (spec §4a)

**Files:**
- Modify: `auto_infer/engine/request.py`, `auto_infer/engine/engine_core.py`, `auto_infer/engine/scheduler.py`, `auto_infer/engine/executor.py` (MockExecutor)
- Test: `tests/test_request.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Request.num_prefill_tokens: int` defaulting to `len(prompt_token_ids)`; every "prompt done" comparison in scheduler/engine uses `num_prefill_tokens` instead of `num_prompt_tokens`. MockExecutor samples from `all_token_ids[num_prefill_tokens - 1]` when prefill completes.

- [ ] **Step 1: Write failing test**

Add to `tests/test_request.py`:

```python
from auto_infer.engine.request import Request, SamplingParams


def test_num_prefill_tokens_defaults_to_prompt_len():
    r = Request(request_id="a", prompt_token_ids=[1, 2, 3],
                sampling=SamplingParams())
    assert r.num_prefill_tokens == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_request.py::test_num_prefill_tokens_defaults_to_prompt_len -v`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

In `request.py`, add a field and a `__post_init__`:

```python
    num_prefill_tokens: int = -1     # tokens to prefill before decode (recompute sets this)

    def __post_init__(self):
        if self.num_prefill_tokens < 0:
            self.num_prefill_tokens = len(self.prompt_token_ids)
```

In `engine_core.py`, in `_step_sync`, `_advance_optimistic`, and `_finalize`, replace every `req.num_prompt_tokens` used as the prompt-done boundary with `req.num_prefill_tokens`. Specifically:
- `_step_sync`: `if req.num_computed_tokens >= req.num_prefill_tokens and ...`
- `_advance_optimistic`: `prompt_done = req.num_computed_tokens + sr.num_tokens_to_compute >= req.num_prefill_tokens`

In `scheduler.py` prefill loop, change `remaining = r.num_prompt_tokens - r.num_computed_tokens` to `remaining = r.num_prefill_tokens - r.num_computed_tokens`. (Prefix-match/allocate in Task 2 still key on the real prompt length — leave those as `num_prompt_tokens`.)

In `executor.py` `MockExecutor.submit`, make prefill-complete sampling recompute-aware:

```python
        for sr in sched_output.scheduled:
            req = scheduler.get_request(sr.request_id)
            start = req.num_computed_tokens
            if start + sr.num_tokens_to_compute >= req.num_prefill_tokens:
                if start >= req.num_prefill_tokens:
                    last = int(prev_sampled[sr.request_id])          # decode
                else:
                    last = req.all_token_ids[req.num_prefill_tokens - 1]  # (re)prefill done
                sampled[sr.request_id] = (last + 1) % self.vocab_size
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_request.py tests/test_engine_core.py -q`
Expected: PASS (normal path: `num_prefill_tokens == num_prompt_tokens`, MockExecutor tokens unchanged — `all_token_ids[num_prompt_tokens-1] == prompt_token_ids[-1]`).

- [ ] **Step 5: Commit**

```bash
git add auto_infer/engine/request.py auto_infer/engine/engine_core.py \
        auto_infer/engine/scheduler.py auto_infer/engine/executor.py tests/test_request.py
git commit -m "feat(engine): num_prefill_tokens recompute boundary (spec §4a)"
```

---

## Task 8: Scheduler preemption — `preempt_one` + `needs_preemption` (spec §4b)

**Files:**
- Modify: `auto_infer/engine/scheduler.py` (`SchedulerOutput`, decode loop, `preempt_one`)
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `Request.num_prefill_tokens` (Task 7), `KVCacheManager.num_free_blocks` (Task 1).
- Produces:
  - `SchedulerOutput.needs_preemption: bool = False`, set when a decode step needs a new block but `kv.num_free_blocks() < 1`.
  - `Scheduler.preempt_one() -> bool`: LIFO-evicts the most-recently-admitted running request (frees its blocks, `num_computed_tokens=0`, `num_prefill_tokens=num_tokens`, moves to waiting front). Returns False if nothing to preempt.

- [ ] **Step 1: Write failing test**

Add to `tests/test_scheduler.py`:

```python
def test_needs_preemption_when_decode_cannot_grow():
    cfg = SchedulerConfig(max_num_batched_tokens=64)
    kv = KVCacheManager(num_blocks=1, block_size=4)      # 1 block total
    s = Scheduler(cfg, kv)
    r = req("a", 4, max_tokens=10)
    s.add_request(r)
    s.schedule()                                          # prefill uses the only block
    r.num_computed_tokens = 4; r.status = RequestStatus.RUNNING
    r.append_output_token(9)                              # now 5 tokens -> needs 2nd block
    s.running = [r]; s.waiting = []
    out = s.schedule()
    assert out.needs_preemption is True


def test_preempt_one_recycles_victim():
    cfg = SchedulerConfig(max_num_batched_tokens=64)
    kv = KVCacheManager(num_blocks=100, block_size=4)
    s = Scheduler(cfg, kv)
    r = req("a", 4, max_tokens=10)
    s.add_request(r); s.schedule()
    r.num_computed_tokens = 4; r.status = RequestStatus.RUNNING
    r.append_output_token(9); s.running = [r]; s.waiting = []
    assert s.preempt_one() is True
    assert r in s.waiting and r not in s.running
    assert r.num_computed_tokens == 0
    assert r.num_prefill_tokens == 5                      # prompt(4) + generated(1)
    assert "a" not in s.block_tables
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_scheduler.py -k "preempt or needs_preemption" -v`
Expected: FAIL (`needs_preemption`/`preempt_one` missing).

- [ ] **Step 3: Implement**

In `scheduler.py`, add the field to `SchedulerOutput`:

```python
@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest]
    num_batched_tokens: int
    needs_preemption: bool = False
```

In the decode loop of `schedule()`, guard the append and set the flag:

```python
        needs_preempt = False
        for r in self.running:
            if num_seqs >= self.config.max_num_seqs or budget < 1:
                break
            if len(r.output_token_ids) >= r.sampling.max_tokens:
                continue
            bt = self.block_tables[r.request_id]
            grows = self.kv.blocks_needed(r.num_tokens) > len(bt)
            if grows and self.kv.num_free_blocks() < 1:
                needs_preempt = True
                break
            self.kv.append_slots(bt, r.num_tokens - 1, 1)
            scheduled.append(ScheduledRequest(r.request_id, 1, False, list(bt)))
            budget -= 1
            num_seqs += 1
```

At the `return`, thread the flag:

```python
        used = self.config.max_num_batched_tokens - budget
        return SchedulerOutput(scheduled, used, needs_preemption=needs_preempt)
```

Add the method:

```python
    def preempt_one(self) -> bool:
        """Recompute-style LIFO preemption: evict the most recently admitted
        running request, free its KV, and requeue it for full recompute."""
        if not self.running:
            return False
        victim = self.running[-1]
        self.kv.free(self.block_tables.pop(victim.request_id, []))
        victim.num_computed_tokens = 0
        victim.num_prefill_tokens = victim.num_tokens        # prompt + generated so far
        victim.status = RequestStatus.WAITING
        self.running.remove(victim)
        self.waiting.insert(0, victim)
        return True
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_scheduler.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add auto_infer/engine/scheduler.py tests/test_scheduler.py
git commit -m "feat(sched): recompute preemption + needs_preemption signal (spec §4b)"
```

---

## Task 9: Engine drain-then-preempt (spec §4c)

**Files:**
- Modify: `auto_infer/engine/engine_core.py` (`_step_sync`, `_step_async`)
- Test: `tests/test_engine_core.py`

**Interfaces:**
- Consumes: `SchedulerOutput.needs_preemption`, `Scheduler.preempt_one` (Task 8), recompute-aware MockExecutor (Task 7).
- Produces: engine that, under memory pressure, drains the in-flight queue before preempting (async) and preempts inline (sync), producing identical output tokens to an unpressured run.

- [ ] **Step 1: Write failing test**

Add to `tests/test_engine_core.py`:

```python
from auto_infer.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.entrypoints.llm import LLM


def _llm(num_blocks, async_scheduling=True):
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
        async_scheduling=async_scheduling,
    )
    return LLM(cfg, executor=MockExecutor(vocab_size=1000))


def test_preemption_matches_unpressured_output():
    prompts = [[1, 2, 3], [4, 5, 6]]
    roomy = _llm(num_blocks=100).generate([list(p) for p in prompts], max_tokens=8)
    tight = _llm(num_blocks=2).generate([list(p) for p in prompts], max_tokens=8)
    assert tight == roomy                       # recompute preserves token stream


def test_preemption_sync_path():
    prompts = [[1, 2, 3], [4, 5, 6]]
    roomy = _llm(num_blocks=100, async_scheduling=False).generate(
        [list(p) for p in prompts], max_tokens=8)
    tight = _llm(num_blocks=2, async_scheduling=False).generate(
        [list(p) for p in prompts], max_tokens=8)
    assert tight == roomy
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_engine_core.py -k preemption -v`
Expected: FAIL (currently raises `MemoryError` in `append_slots`, or hangs without preemption handling).

- [ ] **Step 3: Implement**

In `engine_core.py` `_step_sync`, preempt before executing when signaled:

```python
    def _step_sync(self) -> list[Request]:
        sched_output = self.scheduler.schedule()
        while sched_output.needs_preemption:
            if not self.scheduler.preempt_one():
                break
            sched_output = self.scheduler.schedule()
        sampled = self.executor.execute(sched_output, self.scheduler)
        ...  # rest unchanged
```

In `_step_async`, drain the queue on pressure, then preempt only when empty:

```python
    def _step_async(self) -> list[Request]:
        depth = max(1, self.config.async_batches)
        while len(self._queue) < depth and self._schedulable():
            sched = self.scheduler.schedule()
            if sched.needs_preemption:
                if self._queue:
                    break                          # drain in-flight first (below)
                if not self.scheduler.preempt_one():
                    break
                continue                           # retry with freed space
            if not sched.scheduled:
                break
            handle = self.executor.submit(sched, self.scheduler, self._sampled)
            self._sampled = self.executor.sampled_of(handle)
            self._advance_optimistic(sched)
            self._queue.append((sched, handle))
        if not self._queue:
            return []
        sched_old, handle_old = self._queue.popleft()
        sampled = self.executor.collect(handle_old)
        return self._finalize(sched_old, sampled)
```

Invariant preserved: `preempt_one` runs only when `self._queue` is empty, so no freed block is referenced by an in-flight batch.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_engine_core.py -q`
Expected: PASS (both async and sync preemption match the roomy run).

- [ ] **Step 5: Full host suite**

Run: `pytest -q`
Expected: PASS (all existing + new tests).

- [ ] **Step 6: Commit**

```bash
git add auto_infer/engine/engine_core.py tests/test_engine_core.py
git commit -m "feat(engine): drain-then-preempt async, inline preempt sync (spec §4c)"
```

---

## Task 10: NPU verification scripts (spec §5b) — npu2

**Files:**
- Create: `scripts/verify_prefix_cache.py`, `scripts/verify_preemption.py`

**Interfaces:**
- Consumes: `PagedNpuExecutor`, `LLM`, `EngineConfig` (existing entrypoints; follow the pattern in `scripts/smoke_engine_deepseek_paged.py` / `scripts/smoke_qwen2.py`).

- [ ] **Step 1: Confirm npu2 access** (interactive — do not guess)

Ask the user for the npu2 connection method (ssh alias / working dir / device index / model path already staged there). Do NOT hardcode a path until confirmed. README notes npu2 exists (RoCE NIC down; single-node is sufficient).

- [ ] **Step 2: Write `scripts/verify_prefix_cache.py`**

```python
"""Prefix-cache hit: run a prompt, finish it, then run the SAME prompt again and
assert the second request's scheduled prefill tokens drop to the uncached tail.
Run on npu2: python scripts/verify_prefix_cache.py <model_path>"""
import sys
from auto_infer.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from auto_infer.engine.executor import  # (import the model-path executor used by smoke_qwen2)
# ... construct LLM with enable_prefix_caching=True, block_size matching smoke scripts.
# 1) generate on a long shared prompt to completion (registers blocks on free)
# 2) generate the same prompt; instrument scheduler to print num_computed_tokens at admit
# 3) assert second admit's num_computed_tokens == (len(prompt)-1)//block_size*block_size
```
Fill in using the exact executor/import lines from `scripts/smoke_qwen2.py` (matched at implementation time on npu2, where the model path is known).

- [ ] **Step 3: Write `scripts/verify_preemption.py`**

```python
"""Trigger real preemption with a small num_blocks + concurrent requests, and
assert the generated tokens equal a roomy (no-preemption) run.
Run on npu2: python scripts/verify_preemption.py <model_path>"""
# roomy = LLM(num_blocks=large).generate(prompts, max_tokens=N)
# tight = LLM(num_blocks=small_enough_to_force_preemption).generate(prompts, max_tokens=N)
# assert tight == roomy ; print executor/scheduler preemption count if instrumented
```

- [ ] **Step 4: Run regression + new scripts on npu2**

Run (on npu2):
```bash
pytest -q                                  # host suite
python scripts/smoke_qwen2.py <model>      # regression
python scripts/verify_qwen2_graphdecode_batched.py
python scripts/verify_prefix_cache.py <model>
python scripts/verify_preemption.py <model>
```
Expected: host suite green; smoke outputs unchanged; prefix-cache script shows reduced prefill on the 2nd request; preemption script prints `tight == roomy`.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_prefix_cache.py scripts/verify_preemption.py
git commit -m "test(npu): prefix-cache hit + preemption equivalence scripts (spec §5b)"
```

---

## Self-Review

**Spec coverage:**
- §1a evictable pool → Task 1 ✓ · §1b match-on-admit → Task 2 ✓ · §1c register → Task 2 ✓
- §2a SamplingParams → Task 3 ✓ · §2b vectorized sampler → Task 4 ✓ · §2c wiring → Task 5 ✓
- §3 priority + prefill cap → Task 6 ✓
- §4a num_prefill_tokens → Task 7 ✓ · §4b preempt_one/needs_preemption → Task 8 ✓ · §4c drain-then-preempt → Task 9 ✓
- §5a host tests → Tasks 1–9 ✓ · §5b NPU scripts → Task 10 ✓

**Type consistency:** `num_free_blocks` (free+cached) used consistently in Tasks 1/8. `needs_preemption` field name identical in Tasks 8/9. `num_prefill_tokens` introduced in Task 7 and consumed in Tasks 5/8/9 (Task 5 notes the temporary fallback if run before Task 7). `SamplingTensors` field names identical across Tasks 4/5. `build_sampling_tensors(reqs, vocab, device) -> (SamplingTensors, order)` identical in Tasks 5 definition and both executor call sites.

**Placeholder scan:** Task 10 scripts intentionally defer exact import/model-path lines to implementation-time-on-npu2 (Step 1 gate) — flagged explicitly, not a silent TODO. All host-testable tasks (1–9) carry complete code.

**Ordering note:** Task 5 depends on Task 7's `num_prefill_tokens`; if executed strictly in order (5 before 7), use the documented `num_prompt_tokens` fallback, then flip in Task 7.
