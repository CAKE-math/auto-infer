# Prefix-Cache Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display real prefix-cache queried blocks, hit blocks, and hit rate on the online Prometheus endpoint.

**Architecture:** KVCacheManager remains the sole counter owner. EngineService publishes a thread-safe immutable snapshot, AsyncEngine delegates it, and ServingMetrics renders the snapshot as gauges.

**Tech Stack:** Python, prometheus_client, pytest, FastAPI/httpx tests.

## Global Constraints

- Do not add accounting to the serving request path.
- Do not change the existing `load_snapshot` tuple.
- Define hit rate over eligible full prompt blocks.
- Scraping metrics must not touch mutable KV internals directly.

---

### Task 1: Prefix-cache snapshot and metrics

**Files:**
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_text_serving_api.py`
- Modify: `auto_infer/serving/service.py`
- Modify: `auto_infer/serving/async_engine.py`
- Modify: `auto_infer/serving/metrics.py`
- Modify: `auto_infer/serving/api_server.py`

**Interfaces:**
- Consumes: `KVCacheManager.prefix_queried_blocks`, `KVCacheManager.prefix_hit_blocks`
- Produces: `EngineService.prefix_cache_snapshot -> tuple[int, int]`
- Produces: `AsyncEngine.prefix_cache_snapshot -> tuple[int, int]`
- Produces: `ServingMetrics.set_prefix_cache(*, queried_blocks: int, hit_blocks: int) -> None`

- [ ] **Step 1: Write failing tests**

Add tests asserting exact Prometheus samples for zero queries and for
`queried_blocks=8, hit_blocks=3`; assert service and AsyncEngine snapshot
delegation; assert `/metrics` contains all three values.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
pytest -q tests/test_metrics.py tests/test_text_serving_api.py
```

Expected: failures because the snapshot and metrics do not exist.

- [ ] **Step 3: Implement the minimal data path**

Cache `(queried_blocks, hit_blocks)` in `EngineService._refresh_load_snapshot`,
delegate it from `AsyncEngine`, add three gauges and validated rate calculation
to `ServingMetrics`, and update them in the `/metrics` handler.

- [ ] **Step 4: Verify focused and full suites**

Run:

```bash
pytest -q tests/test_metrics.py tests/test_text_serving_api.py
pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Verify on npu2**

Run the existing BF16 prefix-cache equivalence script and serving-related tests
inside the Ascend container. Expected: cache hit/output parity pass and the
Prometheus tests pass.

- [ ] **Step 6: Commit**

```bash
git add auto_infer/serving tests docs/superpowers
git commit -m "fix: expose prefix cache hit metrics"
```

