import itertools
import queue
import threading
import time

from auto_infer.engine.engine_core import EngineCore
from auto_infer.engine.request import Request
from auto_infer.serving.broker import RequestBroker


class EngineQueueFull(RuntimeError):
    pass


class EngineService:
    """Persistent, single-owner EngineCore service with explicit lifecycle."""

    def __init__(self, config, executor, *, inbox_capacity: int | None = None,
                 close_timeout_s: float = 5.0):
        self.config = config
        self.executor = executor
        self.engine = EngineCore(config, executor)
        self.broker = RequestBroker()
        if inbox_capacity is None:
            inbox_capacity = 2 * config.scheduler.max_num_seqs
        if inbox_capacity <= 0:
            raise ValueError("inbox_capacity must be > 0")
        self._inbox: queue.Queue = queue.Queue(maxsize=inbox_capacity)
        self._abort_lock = threading.Lock()
        self._pending_aborts: set[str] = set()
        self._queued_ids: set[str] = set()
        self._cancelled_queued: set[str] = set()
        self._emitted: dict[str, int] = {}
        self._stage_started: dict[str, tuple[float, float]] = {}
        self._first_stage: dict[str, tuple[float, float]] = {}
        self._decode_stages: dict[str, list[float]] = {}
        self._last_emission_at: dict[str, float] = {}
        self._timing_lock = threading.Lock()
        self._counter = itertools.count()
        self._stopping = threading.Event()
        self._close_lock = threading.Lock()
        self._close_timeout_s = close_timeout_s
        self._closed = False
        self._load_snapshot = (0, 0, 0.0)
        self._prefix_cache_snapshot = (0, 0)
        self.thread = threading.Thread(
            target=self._run, name="AutoInferEngineService", daemon=True)
        self.thread.start()

    @property
    def healthy(self) -> bool:
        return not self._closed and not self._stopping.is_set()

    @property
    def max_kv_tokens(self) -> int:
        return self.config.cache.num_blocks * self.config.cache.block_size

    @property
    def load_snapshot(self) -> tuple[int, int, float]:
        return self._load_snapshot

    @property
    def prefix_cache_snapshot(self) -> tuple[int, int]:
        return self._prefix_cache_snapshot

    def submit(self, ids, sampling):
        rid = f"r{next(self._counter)}"
        stream = self.broker.create(rid)
        with self._abort_lock:
            self._queued_ids.add(rid)
        try:
            self._inbox.put_nowait(
                (rid, tuple(ids), sampling, time.monotonic())
            )
        except queue.Full:
            with self._abort_lock:
                self._queued_ids.discard(rid)
            self.broker.cancel(rid)
            raise EngineQueueFull("engine submission queue is full")
        return rid, stream

    def submit_to(self, ids, sampling, sink) -> str:
        rid = f"r{next(self._counter)}"
        self.broker.register(rid, sink)
        with self._abort_lock:
            self._queued_ids.add(rid)
        try:
            self._inbox.put_nowait(
                (rid, tuple(ids), sampling, time.monotonic())
            )
        except queue.Full:
            with self._abort_lock:
                self._queued_ids.discard(rid)
            self.broker.cancel(rid)
            raise EngineQueueFull("engine submission queue is full")
        return rid

    def take_first_stage_timing(self, request_id: str
                                ) -> tuple[float, float] | None:
        with self._timing_lock:
            return self._first_stage.pop(request_id, None)

    def take_decode_stage_timings(self, request_id: str) -> tuple[float, ...]:
        with self._timing_lock:
            return tuple(self._decode_stages.pop(request_id, ()))

    def release(self, request_id: str) -> None:
        self.broker.cancel(request_id)
        with self._abort_lock:
            if request_id in self._queued_ids:
                self._cancelled_queued.add(request_id)
            else:
                self._pending_aborts.add(request_id)

    def _drain_aborts(self) -> None:
        with self._abort_lock:
            aborts = self._pending_aborts
            self._pending_aborts = set()
        for rid in aborts:
            self.engine.abort(rid)
            self._emitted.pop(rid, None)
            with self._timing_lock:
                self._stage_started.pop(rid, None)
                self._first_stage.pop(rid, None)
                self._decode_stages.pop(rid, None)
                self._last_emission_at.pop(rid, None)

    def _refresh_load_snapshot(self) -> None:
        scheduler = self.engine.scheduler
        kv = self.engine.kv
        utilization = (
            1.0 - kv.num_free_blocks() / kv.num_blocks
            if kv.num_blocks else 0.0
        )
        self._load_snapshot = (
            len(scheduler.running), len(scheduler.waiting), utilization
        )
        self._prefix_cache_snapshot = (
            kv.prefix_queried_blocks, kv.prefix_hit_blocks
        )

    def _collect_control(self):
        with self._abort_lock:
            aborts = self._pending_aborts
            self._pending_aborts = set()
        submits = []
        try:
            for _ in range(self._inbox.maxsize):
                submits.append(self._inbox.get_nowait())
        except queue.Empty:
            pass
        return submits, aborts

    def _apply_control(self, submits, aborts) -> None:
        for rid in aborts:
            self.engine.abort(rid)
            self._emitted.pop(rid, None)
            with self._timing_lock:
                self._stage_started.pop(rid, None)
                self._first_stage.pop(rid, None)
                self._decode_stages.pop(rid, None)
                self._last_emission_at.pop(rid, None)
        for rid, ids, sampling, submitted_at in submits:
            with self._abort_lock:
                self._queued_ids.discard(rid)
                cancelled = rid in self._cancelled_queued
                self._cancelled_queued.discard(rid)
            if cancelled:
                continue
            try:
                self.engine.add_request(Request(rid, list(ids), sampling))
            except Exception as error:
                self.broker.fail(rid, error)
            else:
                self._emitted[rid] = 0
                with self._timing_lock:
                    self._stage_started[rid] = (
                        submitted_at, time.monotonic()
                    )
        self._drain_aborts()
        self._refresh_load_snapshot()

    def _recover(self, error: BaseException) -> None:
        self.broker.fail_all(error)
        self._emitted.clear()
        with self._timing_lock:
            self._stage_started.clear()
            self._first_stage.clear()
            self._decode_stages.clear()
            self._last_emission_at.clear()
        if not self.executor.recoverable:
            self._stopping.set()
            self.broker.close()
            return
        self.engine = EngineCore(self.config, self.executor)

    def _handle_step_error(self, error: BaseException) -> None:
        self._recover(error)

    def _emit_outputs(self, finished) -> None:
        finished_by_id = {req.request_id: req for req in finished}
        for rid in list(self._emitted):
            req = (self.engine.scheduler.get_request_or_none(rid)
                   or finished_by_id.get(rid))
            if req is None:
                continue
            start = self._emitted[rid]
            finalized = (len(req.output_token_ids)
                         if rid in finished_by_id
                         else self.engine.finalized_output_count(rid))
            tokens = tuple(req.output_token_ids[start:finalized])
            if tokens:
                emitted_at = time.monotonic()
                if start == 0:
                    with self._timing_lock:
                        timing = self._stage_started.pop(rid, None)
                        if timing is not None:
                            submitted_at, admitted_at = timing
                            self._first_stage[rid] = (
                                admitted_at - submitted_at,
                                time.monotonic() - admitted_at,
                            )
                        self._last_emission_at[rid] = emitted_at
                else:
                    with self._timing_lock:
                        previous = self._last_emission_at.get(rid)
                        if previous is not None:
                            self._decode_stages.setdefault(rid, []).append(
                                emitted_at - previous
                            )
                        self._last_emission_at[rid] = emitted_at
                self.broker.emit(rid, tokens)
            self._emitted[rid] = finalized
            if rid in finished_by_id:
                self.broker.finish(rid)
                self._emitted.pop(rid, None)
        self._refresh_load_snapshot()

    def _run(self) -> None:
        while not self._stopping.is_set():
            submits, aborts = self._collect_control()
            self._apply_control(submits, aborts)
            if not self.engine.has_unfinished():
                self._stopping.wait(0.0005)
                continue
            try:
                finished = self.engine.step()
            except Exception as error:
                self._handle_step_error(error)
            else:
                self._emit_outputs(finished)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stopping.set()
        self.broker.close()
        self.thread.join(timeout=self._close_timeout_s)
        close_error = self._close_executor_bounded()
        if self.thread.is_alive():
            self.thread.join(timeout=self._close_timeout_s)
        failures = []
        if close_error is not None:
            failures.append(close_error)
        if self.thread.is_alive():
            failures.append("engine service thread did not stop")
        if failures:
            raise RuntimeError("; ".join(failures))

    def _close_executor_bounded(self) -> str | None:
        done = threading.Event()
        errors: list[BaseException] = []

        def close_executor():
            try:
                self.executor.close()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        closer = threading.Thread(
            target=close_executor,
            name="AutoInferExecutorClose",
            daemon=True,
        )
        closer.start()
        if not done.wait(self._close_timeout_s):
            return "executor close timed out"
        if errors:
            return f"executor close failed: {errors[0]}"
        return None
