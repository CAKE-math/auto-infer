import queue
import time

import pytest

from auto_infer.serving.tp_control import (
    ControlAck,
    ControlBatch,
    QueueControlFollower,
    QueueControlLeader,
    ReplicaFatal,
)


def _queues():
    return queue.Queue(), queue.Queue()


def _await_ack(status, sequence):
    ack = status.get(timeout=1.0)
    assert ack.sequence == sequence
    return ack


def test_control_messages_are_immutable():
    batch = ControlBatch(sequence=1, apply_epoch=3)
    ack = ControlAck(rank=1, sequence=1)
    fatal = ReplicaFatal(rank=1, phase="execute", message="boom")

    with pytest.raises(AttributeError):
        batch.sequence = 2
    with pytest.raises(AttributeError):
        ack.error = "late"
    with pytest.raises(AttributeError):
        fatal.phase = "control"


def test_follower_stores_contiguous_delivery_before_acknowledging():
    incoming, status = _queues()
    follower = QueueControlFollower(1, incoming, status)
    try:
        incoming.put(ControlBatch(sequence=1, apply_epoch=4, aborts=("old",)))

        assert _await_ack(status, 1) == ControlAck(rank=1, sequence=1)
        assert follower.pending(3) == ()
        assert follower.pending(4) == (
            ControlBatch(sequence=1, apply_epoch=4, aborts=("old",)),
        )
    finally:
        follower.close()


def test_pending_is_nonblocking_when_idle():
    incoming, status = _queues()
    follower = QueueControlFollower(1, incoming, status)
    try:
        started = time.monotonic()
        assert follower.pending(100) == ()
        assert time.monotonic() - started < 0.1
    finally:
        follower.close()


def test_idle_epoch_batch_is_available_immediately_after_delivery():
    incoming, status = _queues()
    follower = QueueControlFollower(2, incoming, status)
    try:
        batch = ControlBatch(sequence=1, apply_epoch=0, submits=(("r0",),))
        incoming.put(batch)

        _await_ack(status, 1)

        assert follower.pending(0) == (batch,)
    finally:
        follower.close()


def test_follower_acknowledges_shutdown_and_stops_receiver():
    incoming, status = _queues()
    follower = QueueControlFollower(1, incoming, status)
    batch = ControlBatch(sequence=1, apply_epoch=0, shutdown=True)

    incoming.put(batch)

    assert _await_ack(status, 1) == ControlAck(rank=1, sequence=1)
    assert follower.pending(0) == (batch,)
    follower.close()
    assert not follower.thread.is_alive()


@pytest.mark.parametrize("bad_sequence", [1, 3])
def test_follower_rejects_duplicate_or_out_of_order_delivery(bad_sequence):
    incoming, status = _queues()
    follower = QueueControlFollower(1, incoming, status)
    try:
        incoming.put(ControlBatch(sequence=1, apply_epoch=0))
        assert _await_ack(status, 1).error is None
        incoming.put(ControlBatch(sequence=bad_sequence, apply_epoch=0))

        rejected = _await_ack(status, bad_sequence)

        assert rejected.rank == 1
        assert rejected.error is not None
        assert "expected sequence 2" in rejected.error
        assert follower.pending(0) == (
            ControlBatch(sequence=1, apply_epoch=0),
        )
    finally:
        follower.close()


def test_leader_publishes_to_every_follower_and_requires_contiguous_sequence():
    status = queue.Queue()
    first_q, second_q = queue.Queue(), queue.Queue()
    followers = [
        QueueControlFollower(1, first_q, status),
        QueueControlFollower(2, second_q, status),
    ]
    leader = QueueControlLeader((first_q, second_q), status)
    batch = ControlBatch(sequence=1, apply_epoch=2)
    try:
        leader.publish(batch)

        assert leader.publish_count == 1
        assert followers[0].pending(2) == (batch,)
        assert followers[1].pending(2) == (batch,)
        with pytest.raises(ValueError, match="expected sequence 2"):
            leader.publish(batch)
    finally:
        for follower in followers:
            follower.close()
