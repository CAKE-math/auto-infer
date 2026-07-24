"""Single-node supervised tensor-parallel production serving."""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TpServingConfig:
    tp_size: int
    devices: tuple[int, ...] | None = None
    mode: str = "paged"
    master_port: int = 29500
    watchdog_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not 2 <= self.tp_size <= 8:
            raise ValueError("tp_size must satisfy 2 <= tp_size <= 8")
        devices = (tuple(range(self.tp_size))
                   if self.devices is None else tuple(self.devices))
        if len(devices) != self.tp_size:
            raise ValueError("devices length must equal tp_size")
        if any(device < 0 for device in devices):
            raise ValueError("devices must be non-negative")
        if len(set(devices)) != len(devices):
            raise ValueError("devices must be unique")
        if self.mode not in {"recompute", "paged", "graph", "graph_mtp"}:
            raise ValueError(f"unsupported TP mode: {self.mode}")
        if self.mode == "graph_mtp":
            raise ValueError(
                f"{self.mode} tensor parallelism is not numerically gated")
        if self.master_port <= 0:
            raise ValueError("master_port must be > 0")
        if self.watchdog_timeout_s <= 0:
            raise ValueError("tp watchdog timeout must be > 0")
        object.__setattr__(self, "devices", devices)


@dataclass(frozen=True)
class ReplicaStatus:
    """A lifecycle signal sent from one worker to the parent supervisor."""

    rank: int
    kind: str
    message: str = ""


def _worker_environment(rank: int, config: TpServingConfig) -> dict[str, str]:
    if rank < 0 or rank >= config.tp_size:
        raise ValueError("rank must satisfy 0 <= rank < tp_size")
    environment = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(config.master_port),
        "WORLD_SIZE": str(config.tp_size),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(
            str(device) for device in config.devices
        ),
        "HCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    if config.mode == "graph":
        environment["HCCL_OP_EXPANSION_MODE"] = "AIV"
        environment["HCCL_DETERMINISTIC"] = "true"
        environment["LCCL_DETERMINISTIC"] = "1"
    return environment


def validate_tp_serving(
    model_path: str, model_package: str | None = None
) -> None:
    """Reject unsupported model families before spawning NPU workers."""

    from auto_infer.models.registry import (
        get_model_class,
        register_package,
    )

    if model_package is not None:
        register_package(model_package, model_path)
    with open(os.path.join(model_path, "config.json")) as config_file:
        architecture = json.load(config_file)["architectures"][0]
    model_class = get_model_class(architecture)
    if not getattr(model_class, "SUPPORTS_TENSOR_PARALLEL", False):
        raise ValueError(
            f"{architecture} does not support tensor parallel execution"
        )


def _terminate_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=0.2)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join(timeout=1.0)


def supervise_replica(
    processes,
    status_queue,
    *,
    watchdog_timeout_s: float,
    poll_interval_s: float = 0.1,
    ready_event=None,
) -> None:
    """Block until clean replica shutdown or fail the whole replica."""

    startup_started = time.monotonic()
    last_seen: dict[int, float] = {}
    ready: set[int] = set()
    stopped: set[int] = set()
    while True:
        try:
            status = status_queue.get(timeout=poll_interval_s)
        except queue.Empty:
            status = None
        now = time.monotonic()
        if status is not None:
            if not isinstance(status, ReplicaStatus):
                _terminate_processes(processes)
                raise RuntimeError(f"unexpected replica status: {status!r}")
            if status.rank < 0 or status.rank >= len(processes):
                _terminate_processes(processes)
                raise RuntimeError(f"invalid replica rank {status.rank}")
            if status.kind == "fatal":
                _terminate_processes(processes)
                detail = f": {status.message}" if status.message else ""
                raise RuntimeError(f"rank {status.rank} failed{detail}")
            if status.kind == "stopped":
                stopped.add(status.rank)
            elif status.kind == "ready":
                ready.add(status.rank)
                last_seen[status.rank] = now
                if ready_event is not None and len(ready) == len(processes):
                    ready_event.set()
            elif status.kind == "heartbeat":
                last_seen[status.rank] = now
            else:
                _terminate_processes(processes)
                raise RuntimeError(
                    f"rank {status.rank} sent unknown status {status.kind!r}"
                )

        for rank, process in enumerate(processes):
            if process.exitcode is None:
                continue
            if rank not in stopped and process.exitcode != 0:
                _terminate_processes(processes)
                raise RuntimeError(
                    f"rank {rank} exited with code {process.exitcode}"
                )
            if rank not in stopped and any(
                sibling.is_alive() for sibling in processes
            ):
                _terminate_processes(processes)
                raise RuntimeError(
                    f"rank {rank} exited before replica shutdown"
                )

        if (len(ready) < len(processes)
                and now - startup_started > watchdog_timeout_s):
            missing = sorted(set(range(len(processes))) - ready)
            _terminate_processes(processes)
            raise RuntimeError(
                f"replica startup timed out waiting for ranks {missing}"
            )

        expired = [
            rank
            for rank, timestamp in last_seen.items()
            if rank not in stopped and now - timestamp > watchdog_timeout_s
        ]
        if expired:
            _terminate_processes(processes)
            raise RuntimeError(f"rank {expired[0]} watchdog timed out")

        if len(stopped) == len(processes) and all(
            not process.is_alive() for process in processes
        ):
            return


def _service_heartbeat(
    service_box,
    rank: int,
    status_queue,
    stop: threading.Event,
    interval_s: float,
) -> None:
    while not stop.is_set():
        service = service_box.get("service")
        if service is None:
            status_queue.put(ReplicaStatus(rank, "heartbeat"))
            stop.wait(interval_s)
            continue
        fatal = getattr(service, "_fatal_error", None)
        if fatal is not None:
            status_queue.put(ReplicaStatus(rank, "fatal", str(fatal)))
            return
        if not service.thread.is_alive() and not service._stopping.is_set():
            status_queue.put(
                ReplicaStatus(rank, "fatal", "engine service thread stopped")
            )
            return
        status_queue.put(ReplicaStatus(rank, "heartbeat"))
        stop.wait(interval_s)


def _run_worker(rank: int, launch: TpServingConfig, follower_queues,
                control_status_queue, replica_status_queue, ready_event,
                kwargs) -> None:
    os.environ.update(_worker_environment(rank, launch))
    service = None
    service_box = {}
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_service_heartbeat,
        args=(
            service_box,
            rank,
            replica_status_queue,
            heartbeat_stop,
            min(5.0, launch.watchdog_timeout_s / 3),
        ),
        name=f"AutoInferReplicaHeartbeat-{rank}",
        daemon=True,
    )
    heartbeat.start()
    failed = False
    try:
        from auto_infer.config import ParallelConfig
        from auto_infer.engine.factory import build_executor
        from auto_infer.serving.api_server import (
            build_engine_config,
            build_runtime,
            run_runtime,
        )
        from auto_infer.serving.async_engine import AsyncEngine
        from auto_infer.serving.config import ServingConfig
        from auto_infer.serving.tp_control import (
            QueueControlFollower,
            QueueControlLeader,
        )
        from auto_infer.serving.tp_service import SpmdEngineService

        engine_config = build_engine_config(
            model_path=kwargs["model_path"],
            model_package=kwargs["model_package"],
            device_index=rank,
            mode=kwargs["mode"],
            max_model_len=kwargs["max_model_len"],
            num_blocks=kwargs["num_blocks"],
            block_size=kwargs["block_size"],
            max_num_seqs=kwargs["max_num_seqs"],
            max_num_batched_tokens=kwargs["max_num_batched_tokens"],
            max_gear=kwargs["max_gear"],
            max_prefill_tokens=kwargs["max_prefill_tokens"],
            num_speculative_tokens=kwargs["num_speculative_tokens"],
            parallel=ParallelConfig(tp_size=launch.tp_size),
        )
        if rank == 0:
            control = QueueControlLeader(
                follower_queues, control_status_queue,
                ack_timeout_s=launch.watchdog_timeout_s,
            )
        else:
            control = QueueControlFollower(
                rank, follower_queues[rank - 1], control_status_queue
            )
        serving_config = kwargs["serving_config"] or ServingConfig(
            max_num_seqs=kwargs["max_num_seqs"]
        )
        service = SpmdEngineService(
            engine_config,
            build_executor(engine_config),
            rank=rank,
            control=control,
            inbox_capacity=serving_config.max_waiting_requests,
            close_timeout_s=launch.watchdog_timeout_s,
            admission_wait_s=serving_config.admission_wait_ms / 1000.0,
        )
        service_box["service"] = service
        replica_status_queue.put(ReplicaStatus(rank, "ready"))
        while not ready_event.wait(timeout=1.0):
            if heartbeat_stop.is_set():
                return
        if rank == 0:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                kwargs["model_path"], trust_remote_code=True
            )
            runtime = build_runtime(
                tokenizer=tokenizer,
                engine=AsyncEngine.from_service(service),
                model=kwargs["model_path"].rstrip("/").split("/")[-1],
                max_model_len=kwargs["max_model_len"],
                serving_config=serving_config,
            )
            run_runtime(
                runtime,
                host=kwargs["host"],
                port=kwargs["port"],
                access_log=kwargs["access_log"],
            )
        else:
            service.thread.join()
            service.close()
    except BaseException as error:
        failed = True
        replica_status_queue.put(ReplicaStatus(rank, "fatal", str(error)))
        raise
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=1.0)
        if not failed:
            replica_status_queue.put(ReplicaStatus(rank, "stopped"))


def serve_tp(
    model_path: str,
    *,
    tp_size: int,
    devices: tuple[int, ...] | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_package: str | None = None,
    mode: str = "paged",
    max_model_len: int = 4096,
    num_blocks: int = 4096,
    block_size: int = 16,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    max_gear: int = 32,
    max_prefill_tokens: int = 256,
    num_speculative_tokens: int = 1,
    access_log: bool = False,
    serving_config=None,
    master_port: int = 29500,
    watchdog_timeout_s: float = 120.0,
) -> None:
    """Launch and supervise one single-node tensor-parallel replica."""

    import multiprocessing

    launch = TpServingConfig(
        tp_size=tp_size,
        devices=devices,
        mode=mode,
        master_port=master_port,
        watchdog_timeout_s=watchdog_timeout_s,
    )
    validate_tp_serving(model_path, model_package)
    worker_kwargs = {
        "model_path": model_path,
        "host": host,
        "port": port,
        "model_package": model_package,
        "mode": mode,
        "max_model_len": max_model_len,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_gear": max_gear,
        "max_prefill_tokens": max_prefill_tokens,
        "num_speculative_tokens": num_speculative_tokens,
        "access_log": access_log,
        "serving_config": serving_config,
    }
    context = multiprocessing.get_context("spawn")
    follower_queues = [context.Queue() for _ in range(tp_size - 1)]
    control_status_queue = context.Queue()
    replica_status_queue = context.Queue()
    ready_event = context.Event()
    processes = [
        context.Process(
            target=_run_worker,
            args=(
                rank,
                launch,
                follower_queues,
                control_status_queue,
                replica_status_queue,
                ready_event,
                worker_kwargs,
            ),
            name=f"AutoInferTP-{rank}",
        )
        for rank in range(tp_size)
    ]
    for process in processes:
        process.start()
    try:
        supervise_replica(
            processes,
            replica_status_queue,
            watchdog_timeout_s=watchdog_timeout_s,
            ready_event=ready_event,
        )
    except KeyboardInterrupt:
        _terminate_processes(processes)
    finally:
        for process in processes:
            process.join(timeout=1.0)
