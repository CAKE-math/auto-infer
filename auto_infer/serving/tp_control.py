"""Transport-neutral messages and queue transport for SPMD engine control."""

import queue
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlBatch:
    sequence: int
    apply_epoch: int
    submits: tuple = ()
    aborts: tuple[str, ...] = ()
    shutdown: bool = False

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("sequence must be > 0")
        if self.apply_epoch < 0:
            raise ValueError("apply_epoch must be >= 0")
        object.__setattr__(self, "submits", tuple(self.submits))
        object.__setattr__(self, "aborts", tuple(self.aborts))


@dataclass(frozen=True)
class ControlAck:
    rank: int
    sequence: int
    error: str | None = None


@dataclass(frozen=True)
class ReplicaFatal:
    rank: int
    phase: str
    message: str


class QueueControlLeader:
    """Fan out batches and wait for delivery acknowledgements from followers."""

    def __init__(self, follower_queues, status_queue, *, ack_timeout_s=5.0):
        self._follower_queues = tuple(follower_queues)
        self._status_queue = status_queue
        self._ack_timeout_s = ack_timeout_s
        self._next_sequence = 1
        self.publish_count = 0

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def publish(self, batch: ControlBatch) -> None:
        if batch.sequence != self._next_sequence:
            raise ValueError(
                f"expected sequence {self._next_sequence}, got {batch.sequence}"
            )
        for follower_queue in self._follower_queues:
            follower_queue.put(batch)
        ranks = set()
        while len(ranks) < len(self._follower_queues):
            try:
                status = self._status_queue.get(timeout=self._ack_timeout_s)
            except queue.Empty as error:
                raise TimeoutError(
                    f"timed out waiting for sequence {batch.sequence} delivery"
                ) from error
            if isinstance(status, ReplicaFatal):
                raise RuntimeError(
                    f"rank {status.rank} failed during {status.phase}: "
                    f"{status.message}"
                )
            if not isinstance(status, ControlAck):
                raise RuntimeError(f"unexpected control status: {status!r}")
            if status.sequence != batch.sequence:
                raise RuntimeError(
                    f"unexpected acknowledgement sequence {status.sequence}; "
                    f"expected {batch.sequence}"
                )
            if status.error is not None:
                raise RuntimeError(
                    f"rank {status.rank} rejected sequence {batch.sequence}: "
                    f"{status.error}"
                )
            if status.rank in ranks:
                raise RuntimeError(
                    f"duplicate acknowledgement from rank {status.rank}"
                )
            ranks.add(status.rank)
        self._next_sequence += 1
        self.publish_count += 1

    def report(self, status: ReplicaFatal) -> None:
        self._status_queue.put(status)


class QueueControlFollower:
    """Receive batches off the engine thread and expose due work nonblockingly."""

    def __init__(self, rank, incoming_queue, status_queue):
        self.rank = rank
        self._incoming_queue = incoming_queue
        self._status_queue = status_queue
        self._lock = threading.Lock()
        self._pending: list[ControlBatch] = []
        self._next_sequence = 1
        self._closed = threading.Event()
        self.thread = threading.Thread(
            target=self._receive,
            name=f"AutoInferControlReceiver-{rank}",
            daemon=True,
        )
        self.thread.start()

    def _receive(self) -> None:
        while not self._closed.is_set():
            batch = self._incoming_queue.get()
            if batch is None:
                return
            if not isinstance(batch, ControlBatch):
                continue
            with self._lock:
                expected = self._next_sequence
                if batch.sequence == expected:
                    self._pending.append(batch)
                    self._next_sequence += 1
                    error = None
                else:
                    error = (
                        f"expected sequence {expected}, got {batch.sequence}"
                    )
            self._status_queue.put(
                ControlAck(self.rank, batch.sequence, error)
            )
            if error is None and batch.shutdown:
                return

    def pending(self, epoch: int) -> tuple[ControlBatch, ...]:
        with self._lock:
            split = 0
            for batch in self._pending:
                if batch.apply_epoch > epoch:
                    break
                split += 1
            due = tuple(self._pending[:split])
            del self._pending[:split]
        return due

    def report(self, status: ReplicaFatal) -> None:
        self._status_queue.put(status)

    def close(self) -> None:
        if self._closed.is_set():
            self.thread.join(timeout=1.0)
            return
        self._closed.set()
        if self.thread.is_alive():
            self._incoming_queue.put(None)
        self.thread.join(timeout=1.0)
