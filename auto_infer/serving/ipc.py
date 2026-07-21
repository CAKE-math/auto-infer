"""Persistent process transport with request-id response demultiplexing."""
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestEnvelope:
    request_id: str
    kind: str
    payload: tuple = ()


@dataclass(frozen=True)
class ResponseEnvelope:
    request_id: str
    kind: str
    payload: object = None


def _production_runtime(model_path, device_index):
    from importlib import import_module
    import_module("torch_npu")
    from auto_infer.config import (CacheConfig, EngineConfig, ExecutionConfig,
                                   ModelConfig, SchedulerConfig)
    from auto_infer.engine.factory import build_executor
    config = EngineConfig(
        model=ModelConfig(model_path=model_path),
        cache=CacheConfig(block_size=16, num_blocks=4096),
        scheduler=SchedulerConfig(max_num_batched_tokens=4096),
        execution=ExecutionConfig(mode="paged", device_index=device_index))
    return config, build_executor(config)


def _engine_worker(req_q, resp_q, model_path, device_index, config, executor):
    from auto_infer.engine.request import SamplingParams
    from auto_infer.serving.service import EngineService
    if config is None:
        config, executor = _production_runtime(model_path, device_index)
    service = EngineService(config, executor)
    active = {}
    resp_q.put(ResponseEnvelope("__worker__", "ready", 1))
    stopping = False
    try:
        while not stopping:
            try:
                envelope = req_q.get(timeout=0.001)
            except queue.Empty:
                envelope = None
            if envelope is not None:
                if envelope.kind == "close":
                    stopping = True
                elif envelope.kind == "cancel":
                    item = active.pop(envelope.request_id, None)
                    if item is not None:
                        service.release(item[0])
                elif envelope.kind == "submit":
                    ids, max_tokens = envelope.payload
                    service_id, stream = service.submit(
                        ids, SamplingParams(max_tokens=max_tokens))
                    active[envelope.request_id] = (service_id, stream)
                else:
                    resp_q.put(ResponseEnvelope(
                        envelope.request_id, "error", f"unknown request kind: {envelope.kind}"))
            for request_id, (service_id, stream) in list(active.items()):
                try:
                    token = stream.get_nowait()
                except queue.Empty:
                    continue
                except Exception as error:
                    resp_q.put(ResponseEnvelope(request_id, "error", str(error)))
                    active.pop(request_id, None)
                    continue
                if token is None:
                    resp_q.put(ResponseEnvelope(request_id, "finished"))
                    active.pop(request_id, None)
                else:
                    resp_q.put(ResponseEnvelope(request_id, "token", int(token)))
    finally:
        service.close()
        resp_q.put(ResponseEnvelope("__worker__", "closed"))


class EngineProcess:
    def __init__(self, model_path: str, device_index: int = 0, *,
                 _config=None, _executor=None):
        ctx = mp.get_context("spawn")
        self.req_q = ctx.Queue()
        self.resp_q = ctx.Queue()
        self._streams: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
        self.proc = ctx.Process(
            target=_engine_worker,
            args=(self.req_q, self.resp_q, model_path, device_index,
                  _config, _executor), daemon=True)
        self.proc.start()
        ready = self.resp_q.get(timeout=30)
        if ready.kind != "ready":
            raise RuntimeError(f"engine process failed to start: {ready}")
        self.worker_generation = int(ready.payload)
        self._demux = threading.Thread(
            target=self._demultiplex, name="AutoInferIPCDemux", daemon=True)
        self._demux.start()

    @classmethod
    def for_testing(cls, config, executor):
        return cls("/mock", _config=config, _executor=executor)

    def _demultiplex(self):
        while True:
            envelope = self.resp_q.get()
            if envelope.kind == "closed":
                return
            with self._lock:
                stream = self._streams.get(envelope.request_id)
            if stream is None:
                continue
            if envelope.kind == "token":
                stream.put(envelope.payload)
            elif envelope.kind == "finished":
                stream.put(None)
            elif envelope.kind == "error":
                stream.put(RuntimeError(str(envelope.payload)))

    def generate_stream(self, rid: str, ids: list[int], max_tokens: int):
        if self._closed:
            raise RuntimeError("engine process is closed")
        stream = queue.Queue()
        with self._lock:
            if rid in self._streams:
                raise ValueError(f"duplicate request id: {rid}")
            self._streams[rid] = stream
        self.req_q.put(RequestEnvelope(rid, "submit", (tuple(ids), max_tokens)))
        try:
            while True:
                item = stream.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            with self._lock:
                self._streams.pop(rid, None)
            if not self._closed:
                self.req_q.put(RequestEnvelope(rid, "cancel"))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.req_q.put(RequestEnvelope("__worker__", "close"))
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
            raise RuntimeError("engine process did not stop cleanly")
        self._demux.join(timeout=5)
