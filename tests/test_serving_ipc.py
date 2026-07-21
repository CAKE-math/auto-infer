from concurrent.futures import ThreadPoolExecutor

from auto_infer.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from auto_infer.engine.executor import MockExecutor
from auto_infer.serving.ipc import EngineProcess


def _cfg():
    return EngineConfig(
        model=ModelConfig("/mock"), cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(max_num_batched_tokens=64))


def test_concurrent_ipc_streams_are_demultiplexed():
    process = EngineProcess.for_testing(_cfg(), MockExecutor(vocab_size=1000))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(lambda: list(process.generate_stream("a", [1], 3)))
            b = pool.submit(lambda: list(process.generate_stream("b", [10], 3)))
            assert a.result(timeout=5) == [2, 3, 4]
            assert b.result(timeout=5) == [11, 12, 13]
        assert process.worker_generation == 1
    finally:
        process.close()
    assert not process.proc.is_alive()
