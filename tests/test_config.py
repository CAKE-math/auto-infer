from auto_infer.config import (
    ModelConfig, ParallelConfig, CacheConfig, SchedulerConfig, EngineConfig,
    ExecutionConfig,
)


def test_defaults_and_world_size():
    p = ParallelConfig(tp_size=2, dp_size=2)
    assert p.world_size == 4
    cfg = EngineConfig(model=ModelConfig(model_path="/m"))
    assert cfg.cache.block_size == 16
    assert cfg.scheduler.enable_chunked_prefill is True
    assert cfg.async_scheduling is False   # SP4: sync strictly faster single-card; async opt-in
    assert cfg.execution.max_gear == 32
    assert cfg.execution.max_prefill_tokens == 256


def test_invalid_block_size():
    import pytest
    with pytest.raises(ValueError):
        CacheConfig(block_size=0)


def test_invalid_scheduler_limits():
    import pytest
    with pytest.raises(ValueError, match="max_num_seqs"):
        SchedulerConfig(max_num_seqs=0)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        SchedulerConfig(max_num_batched_tokens=0)


def test_invalid_model_parallel_and_execution_config():
    import pytest
    with pytest.raises(ValueError, match="model_path"):
        ModelConfig(model_path="")
    with pytest.raises(ValueError, match="tp_size"):
        ParallelConfig(tp_size=0)
    with pytest.raises(ValueError, match="node_rank"):
        ParallelConfig(nnodes=1, node_rank=1)
    with pytest.raises(ValueError, match="mode"):
        ExecutionConfig(mode="unknown")
    with pytest.raises(ValueError, match="max_gear"):
        ExecutionConfig(max_gear=0)
    with pytest.raises(ValueError, match="max_prefill_tokens"):
        ExecutionConfig(max_prefill_tokens=0)
    with pytest.raises(ValueError, match="async_batches"):
        EngineConfig(model=ModelConfig("/m"), async_batches=0)


def test_spec_decode_rejects_separate_async_batch_queue():
    from auto_infer.config import SpecDecodeConfig
    import pytest

    with pytest.raises(ValueError, match="speculative decoding.*async_scheduling"):
        EngineConfig(
            model=ModelConfig("/m"), async_scheduling=True,
            spec_decode=SpecDecodeConfig())
