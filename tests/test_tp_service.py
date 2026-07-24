import queue
import threading

import pytest

from auto_infer.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.request import SamplingParams
from auto_infer.serving.tp_control import (
    QueueControlFollower,
    QueueControlLeader,
    ReplicaFatal,
)
from auto_infer.serving.tp_service import SpmdEngineService


def _cfg():
    return EngineConfig(
        model=ModelConfig(model_path="/mock"),
        cache=CacheConfig(block_size=4, num_blocks=100),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=64,
            enable_prefix_caching=True,
        ),
    )


def _services(monkeypatch, *, leader_executor=None):
    incoming = queue.Queue()
    status = queue.Queue()
    follower_control = QueueControlFollower(1, incoming, status)
    leader_control = QueueControlLeader((incoming,), status)
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    follower = SpmdEngineService(
        _cfg(), MockExecutor(vocab_size=1000),
        rank=1, control=follower_control,
    )
    leader = SpmdEngineService(
        _cfg(), leader_executor or MockExecutor(vocab_size=1000),
        rank=0, control=leader_control,
    )
    return leader, follower, leader_control, follower_control, status


def _state(service):
    scheduler = service.engine.scheduler
    requests = {
        rid: (
            request.status,
            tuple(request.output_token_ids),
            request.num_computed_tokens,
        )
        for rid, request in scheduler._requests.items()
    }
    return {
        "request_ids": tuple(sorted(requests)),
        "requests": requests,
        "running": tuple(request.request_id for request in scheduler.running),
        "waiting": tuple(request.request_id for request in scheduler.waiting),
        "prefix": (
            service.engine.kv.prefix_queried_blocks,
            service.engine.kv.prefix_hit_blocks,
        ),
        "block_tables": {
            rid: tuple(table)
            for rid, table in scheduler.block_tables.items()
        },
    }


def _advance_pair(leader, follower):
    leader._run_once()
    follower._run_once()
    assert follower.engine_epoch == leader.engine_epoch
    assert _state(follower) == _state(leader)


def _drain(stream):
    tokens = []
    while True:
        token = stream.get(timeout=1.0)
        if token is None:
            return tokens
        tokens.append(token)


def test_two_services_converge_through_overlapping_submit_abort_and_finish(
    monkeypatch,
):
    leader, follower, transport, follower_control, _ = _services(monkeypatch)
    try:
        first, first_stream = leader.submit(
            [1, 2, 3, 4], SamplingParams(max_tokens=8)
        )
        aborted, aborted_stream = leader.submit(
            [1, 2, 3, 9], SamplingParams(max_tokens=8)
        )
        _advance_pair(leader, follower)

        late, late_stream = leader.submit(
            [1, 2, 3, 4, 5], SamplingParams(max_tokens=6)
        )
        _advance_pair(leader, follower)
        leader.release(aborted)

        for _ in range(20):
            _advance_pair(leader, follower)
            if not leader.engine.has_unfinished():
                break

        assert not leader.engine.has_unfinished()
        assert not follower.engine.has_unfinished()
        assert _drain(first_stream) == list(range(5, 13))
        assert _drain(aborted_stream) == [10, 11]
        assert _drain(late_stream) == list(range(6, 12))
        assert transport.publish_count == 3
        assert leader.last_applied_sequence == follower.last_applied_sequence == 3
        assert first != aborted != late
    finally:
        follower_control.close()


def test_long_decode_without_control_changes_does_not_publish(monkeypatch):
    leader, follower, transport, follower_control, _ = _services(monkeypatch)
    try:
        _, stream = leader.submit([40], SamplingParams(max_tokens=12))
        _advance_pair(leader, follower)
        initial_publish_count = transport.publish_count

        while leader.engine.has_unfinished():
            _advance_pair(leader, follower)

        assert _drain(stream) == list(range(41, 53))
        assert initial_publish_count == 1
        assert transport.publish_count == initial_publish_count
    finally:
        follower_control.close()


def test_future_control_applies_if_active_work_finishes_before_target_epoch(
    monkeypatch,
):
    leader, follower, _, follower_control, _ = _services(monkeypatch)
    try:
        _, active_stream = leader.submit(
            [60], SamplingParams(max_tokens=2)
        )
        _advance_pair(leader, follower)
        _, late_stream = leader.submit(
            [70], SamplingParams(max_tokens=2)
        )

        _advance_pair(leader, follower)
        assert not leader.engine.has_unfinished()
        for _ in range(3):
            _advance_pair(leader, follower)

        assert leader.last_applied_sequence == 2
        assert follower.last_applied_sequence == 2
        assert _drain(active_stream) == [61, 62]
        assert _drain(late_stream) == [71, 72]
    finally:
        follower_control.close()


def test_cancellation_before_control_publication_is_not_sent_to_followers(
    monkeypatch,
):
    leader, follower, transport, follower_control, _ = _services(monkeypatch)
    try:
        request_id, stream = leader.submit(
            [70], SamplingParams(max_tokens=4)
        )
        leader.release(request_id)

        _advance_pair(leader, follower)

        assert _drain(stream) == []
        assert not leader.engine.has_unfinished()
        assert not follower.engine.has_unfinished()
        assert transport.publish_count == 0
    finally:
        follower_control.close()


def test_cancellation_after_publication_is_applied_in_sequence_on_all_ranks(
    monkeypatch,
):
    leader, follower, transport, follower_control, _ = _services(monkeypatch)
    try:
        _, active_stream = leader.submit(
            [80], SamplingParams(max_tokens=8)
        )
        _advance_pair(leader, follower)
        late, late_stream = leader.submit(
            [90], SamplingParams(max_tokens=4)
        )
        _advance_pair(leader, follower)

        leader.release(late)
        while leader.engine.has_unfinished() or leader._leader_pending:
            _advance_pair(leader, follower)

        assert _drain(late_stream) == []
        assert _drain(active_stream) == list(range(81, 89))
        assert transport.publish_count == 3
    finally:
        follower_control.close()


def test_distributed_step_error_is_fatal_and_never_recovers(monkeypatch):
    class BoomExecutor(MockExecutor):
        def execute(self, plan):
            raise RuntimeError("collective failed")

    leader, _, _, follower_control, status = _services(
        monkeypatch, leader_executor=BoomExecutor(vocab_size=1000)
    )
    leader._recover = lambda error: pytest.fail("_recover must not be called")
    original_engine = leader.engine
    _, stream = leader.submit([1, 2], SamplingParams(max_tokens=3))
    try:
        leader._run_once()

        with pytest.raises(
            RuntimeError, match="engine request failed: collective failed"
        ):
            stream.get(timeout=1.0)
        assert leader.engine is original_engine
        assert not leader.healthy
        assert status.get(timeout=1.0) == ReplicaFatal(
            rank=0, phase="execute", message="collective failed"
        )
    finally:
        follower_control.close()
