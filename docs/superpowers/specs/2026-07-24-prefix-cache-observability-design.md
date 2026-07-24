# Prefix-Cache Observability Design

## Goal

Expose the engine's real prefix-cache block counters through the online
Prometheus endpoint without duplicating cache accounting in the serving layer.

## Data Flow

`KVCacheManager` remains the only writer of `prefix_queried_blocks` and
`prefix_hit_blocks`. `EngineService`, which owns the engine thread, copies both
cumulative values into an immutable two-integer snapshot whenever it refreshes
the existing load snapshot. `AsyncEngine` exposes that snapshot read-only.
`/metrics` transfers it to `ServingMetrics`.

The existing three-field `load_snapshot` contract remains unchanged.

## Metrics

- `auto_infer_serving_prefix_cache_queried_blocks`
- `auto_infer_serving_prefix_cache_hit_blocks`
- `auto_infer_serving_prefix_cache_hit_rate`

All values are cumulative for the current engine lifetime. The rate is
`hit_blocks / queried_blocks`, or zero when no block has been queried. A query
means a full prompt block eligible for reuse; partial blocks are not counted.

## Correctness

- Serving never estimates hits from request lengths.
- The snapshot is refreshed only on the engine owner thread.
- Scraping `/metrics` performs no cache lookup and takes no engine lock.
- Disabled or unused prefix caching reports zero for all three metrics.
- Existing cache reuse, logging, and `load_snapshot` behavior remain unchanged.

## Verification

Tests must first fail for the absent metrics, then cover exact values, the
zero-query case, service snapshot refresh, AsyncEngine delegation, and the
actual `/metrics` endpoint.

