import logging

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.metrics import StatLogger
from auto_infer.entrypoints.llm import LLM
from auto_infer.serving.metrics import ServingMetrics


class _KV:  # minimal gauge stub for StatLogger.maybe_log
    num_blocks = 10
    prefix_queried_blocks = 4
    prefix_hit_blocks = 1

    def num_free_blocks(self):
        return 5


def test_statlogger_emits_after_interval_and_resets(caplog):
    sl = StatLogger(interval_s=0.0)
    sl.record_step(prefill_toks=10, gen_toks=3)
    sl.record_ttft(0.05)
    with caplog.at_level(logging.INFO, logger="auto_infer.metrics"):
        sl.maybe_log(now=sl._t0 + 1.0, running=2, waiting=1, kv=_KV(), num_preemptions=0)
    assert any("[engine]" in r.message for r in caplog.records)
    assert sl._prefill_toks == 0 and sl._gen_toks == 0 and sl._ttfts == []   # window reset


def test_statlogger_gated_below_interval():
    sl = StatLogger(interval_s=100.0)
    sl.record_step(5, 1)
    sl.maybe_log(now=sl._t0 + 0.01, running=1, waiting=0, kv=_KV(), num_preemptions=0)
    assert sl._gen_toks == 1        # below interval: not logged, not reset


def test_statlogger_spec_line_only_when_fed(caplog):
    sl = StatLogger(interval_s=0.0)
    sl.record_step(0, 4)
    sl.record_spec(steps=2, accepted=6, accepted_per_position=(2, 2, 2))
    with caplog.at_level(logging.INFO, logger="auto_infer.metrics"):
        sl.maybe_log(now=sl._t0 + 1.0, running=0, waiting=0, kv=_KV(), num_preemptions=0)
    assert "spec-accept" in "\n".join(r.message for r in caplog.records)
    assert sl._spec_accepted_per_position == []


def test_statlogger_accumulates_acceptance_by_draft_position():
    sl = StatLogger(interval_s=100.0)
    sl.record_spec(steps=4, accepted=5, accepted_per_position=(3, 2, 0))
    sl.record_spec(steps=2, accepted=3, accepted_per_position=(2, 1))

    assert sl._spec_steps == 6
    assert sl._spec_accepted_per_position == [5, 3, 0]


def _llm_stats(interval_s=1e9, async_scheduling=False):
    cfg = EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64),
        log_stats=True, log_stats_interval_s=interval_s,
        async_scheduling=async_scheduling,
    )
    return LLM(cfg, executor=MockExecutor(vocab_size=1000))


def test_engine_records_ttft_and_tokens_sync():
    llm = _llm_stats()                # huge interval: window not flushed mid-run
    outs = llm.generate([[1, 2, 3], [10, 11]], max_tokens=4)
    sl = llm.engine.stat_logger
    assert sl._gen_toks == len(outs) * 4             # 2 reqs * 4 tokens each
    assert len(sl._ttfts) == 2                        # one TTFT sample per request
    assert all(t >= 0 for t in sl._ttfts)


def test_engine_records_ttft_and_tokens_async():
    llm = _llm_stats(async_scheduling=True)
    outs = llm.generate([[1, 2, 3], [10, 11]], max_tokens=4)
    sl = llm.engine.stat_logger
    assert sl._gen_toks == len(outs) * 4
    assert len(sl._ttfts) == 2


def test_log_stats_off_by_default():
    cfg = EngineConfig(model=ModelConfig(model_path="/mock"),
                       cache=CacheConfig(block_size=4, num_blocks=100),
                       scheduler=SchedulerConfig(max_num_batched_tokens=64))
    llm = LLM(cfg, executor=MockExecutor(vocab_size=1000))
    assert llm.engine.stat_logger is None            # zero overhead when off
    assert len(llm.generate([[1, 2, 3]], max_tokens=3)[0]) == 3


def test_prometheus_render_has_required_serving_stages():
    metrics = ServingMetrics()
    for stage in (
        "http_parse",
        "admission_wait",
        "tokenize",
        "engine_queue",
        "prefill",
        "decode",
        "detokenize",
        "sse_send",
        "ttft",
        "itl",
        "e2e",
    ):
        metrics.observe_stage(stage, 0.001)

    text = metrics.render()

    for stage in (
        "http_parse",
        "admission_wait",
        "tokenize",
        "engine_queue",
        "prefill",
        "decode",
        "detokenize",
        "sse_send",
        "ttft",
        "itl",
        "e2e",
    ):
        assert f"auto_infer_serving_{stage}_seconds" in text


def test_serving_metrics_use_isolated_registries():
    first = ServingMetrics()
    second = ServingMetrics()

    first.record_rejection("http")

    assert 'stage="http"' in first.render()
    assert 'stage="http"' not in second.render()
