"""Replicated EngineService driven by epoch-tagged change-only control."""

import threading
import time

from auto_infer.serving.service import EngineService
from auto_infer.serving.tp_control import (
    ControlBatch,
    QueueControlFollower,
    QueueControlLeader,
    ReplicaFatal,
)


class SpmdEngineService(EngineService):
    """Run identical EngineCore state on a control leader and its followers."""

    def __init__(self, config, executor, *, rank: int, control,
                 inbox_capacity: int | None = None,
                 close_timeout_s: float = 5.0):
        if rank < 0:
            raise ValueError("rank must be >= 0")
        if rank == 0 and not isinstance(control, QueueControlLeader):
            raise TypeError("rank 0 requires QueueControlLeader")
        if rank != 0 and not isinstance(control, QueueControlFollower):
            raise TypeError("follower rank requires QueueControlFollower")
        self.rank = rank
        self.control = control
        self.engine_epoch = 0
        self.last_applied_sequence = 0
        self._next_sequence = 1
        self._leader_pending: list[ControlBatch] = []
        self._published_submission_ids: set[str] = set()
        self._publish_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        super().__init__(
            config,
            executor,
            inbox_capacity=inbox_capacity,
            close_timeout_s=close_timeout_s,
        )

    def release(self, request_id: str) -> None:
        self.broker.cancel(request_id)
        with self._abort_lock:
            if request_id in self._queued_ids:
                if request_id in self._published_submission_ids:
                    self._pending_aborts.add(request_id)
                else:
                    self._cancelled_queued.add(request_id)
            else:
                self._pending_aborts.add(request_id)

    def _collect_publishable_control(self):
        submits, aborts = self._collect_control()
        publishable = []
        with self._abort_lock:
            for submit in submits:
                request_id = submit[0]
                if request_id in self._cancelled_queued:
                    self._queued_ids.discard(request_id)
                    self._cancelled_queued.discard(request_id)
                    continue
                self._published_submission_ids.add(request_id)
                publishable.append(submit)
        return publishable, aborts

    def _drain_aborts(self) -> None:
        # A newly arriving abort must be published before either rank applies it.
        return

    def _publish(self, submits=(), aborts=(), *, shutdown=False) -> None:
        active = self.engine.has_unfinished()
        apply_epoch = self.engine_epoch + 2 if active else self.engine_epoch
        with self._publish_lock:
            batch = ControlBatch(
                sequence=self._next_sequence,
                apply_epoch=apply_epoch,
                submits=tuple(submits),
                aborts=tuple(sorted(aborts)),
                shutdown=shutdown,
            )
            self.control.publish(batch)
            self._next_sequence += 1
            self._leader_pending.append(batch)

    def _due_batches(self) -> tuple[ControlBatch, ...]:
        idle = not self.engine.has_unfinished()
        if self.rank != 0:
            epoch = float("inf") if idle else self.engine_epoch
            return self.control.pending(epoch)
        split = 0
        for batch in self._leader_pending:
            if not idle and batch.apply_epoch > self.engine_epoch:
                break
            split += 1
        due = tuple(self._leader_pending[:split])
        del self._leader_pending[:split]
        return due

    def _apply_batch(self, batch: ControlBatch) -> None:
        if batch.sequence != self.last_applied_sequence + 1:
            raise RuntimeError(
                f"expected apply sequence {self.last_applied_sequence + 1}, "
                f"got {batch.sequence}"
            )
        self._apply_control(batch.submits, batch.aborts)
        if self.rank == 0:
            with self._abort_lock:
                for submit in batch.submits:
                    self._published_submission_ids.discard(submit[0])
        self.last_applied_sequence = batch.sequence
        if batch.shutdown:
            self._stopping.set()
            self.broker.close()

    def _run_once(self) -> None:
        if self._stopping.is_set():
            return
        try:
            if self.rank == 0:
                submits, aborts = self._collect_publishable_control()
                if submits or aborts:
                    self._publish(submits, aborts)
            batches = self._due_batches()
            if batches and not self.engine.has_unfinished():
                self.engine_epoch = max(
                    self.engine_epoch,
                    max(batch.apply_epoch for batch in batches),
                )
            for batch in batches:
                self._apply_batch(batch)
            if self._stopping.is_set():
                return
            if not self.engine.has_unfinished():
                self._stopping.wait(0.0005)
                return
            finished = self.engine.step()
        except Exception as error:
            self._handle_step_error(error)
        else:
            self.engine_epoch += 1
            self._emit_outputs(finished)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._run_once()

    def _handle_step_error(self, error: BaseException) -> None:
        self._fatal_error = error
        self.broker.fail_all(error)
        self._emitted.clear()
        with self._timing_lock:
            self._stage_started.clear()
            self._first_stage.clear()
            self._decode_stages.clear()
            self._last_emission_at.clear()
        self._stopping.set()
        self.broker.close()
        self.control.report(
            ReplicaFatal(self.rank, "execute", str(error))
        )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        failures = []
        if self.rank == 0 and self._fatal_error is None:
            try:
                submits, aborts = self._collect_publishable_control()
                self._publish(submits, aborts, shutdown=True)
            except Exception as error:
                failures.append(f"control shutdown failed: {error}")
                self._stopping.set()
        else:
            self._stopping.set()
        self.broker.close()
        if self.thread.ident is not None:
            self.thread.join(timeout=self._close_timeout_s)
        elif not self._stopping.is_set():
            deadline = time.monotonic() + self._close_timeout_s
            while not self._stopping.is_set() and time.monotonic() < deadline:
                self._run_once()
        close_error = self._close_executor_bounded()
        if isinstance(self.control, QueueControlFollower):
            self.control.close()
        if close_error is not None:
            failures.append(close_error)
        if self.thread.is_alive():
            failures.append("engine service thread did not stop")
        if failures:
            raise RuntimeError("; ".join(failures))
