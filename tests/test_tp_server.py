import queue
import threading
import json

import pytest

from auto_infer.serving.tp_server import (
    ReplicaStatus,
    TpServingConfig,
    _worker_environment,
    supervise_replica,
    validate_tp_serving,
)


def test_tp_config_defaults_to_consecutive_devices():
    config = TpServingConfig(tp_size=4)

    assert config.devices == (0, 1, 2, 3)
    assert config.master_port == 29500
    assert config.watchdog_timeout_s == 120.0


@pytest.mark.parametrize("tp_size", [0, 1, 9])
def test_tp_config_requires_multiple_ranks(tp_size):
    with pytest.raises(ValueError, match="tp_size"):
        TpServingConfig(tp_size=tp_size)


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ((0,), "length"),
        ((0, 0), "unique"),
        ((0, -1), "non-negative"),
    ],
)
def test_tp_config_rejects_invalid_devices(devices, message):
    with pytest.raises(ValueError, match=message):
        TpServingConfig(tp_size=2, devices=devices)


def test_tp_config_rejects_graph_mtp_until_numerically_gated():
    with pytest.raises(ValueError, match="graph_mtp"):
        TpServingConfig(tp_size=2, mode="graph_mtp")


def test_tp_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        TpServingConfig(tp_size=2, mode="unknown")


def test_tp_config_rejects_non_positive_watchdog():
    with pytest.raises(ValueError, match="watchdog"):
        TpServingConfig(tp_size=2, watchdog_timeout_s=0)


def test_worker_environment_maps_rank_to_selected_device():
    config = TpServingConfig(
        tp_size=2, devices=(4, 6), master_port=29600
    )

    environment = _worker_environment(1, config)

    assert environment["RANK"] == "1"
    assert environment["WORLD_SIZE"] == "2"
    assert environment["LOCAL_RANK"] == "1"
    assert environment["MASTER_ADDR"] == "127.0.0.1"
    assert environment["MASTER_PORT"] == "29600"
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "4,6"
    assert environment["HCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert environment["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"


def test_graph_worker_environment_enables_aiv_expansion():
    environment = _worker_environment(
        0, TpServingConfig(tp_size=2, mode="graph")
    )

    assert environment["HCCL_OP_EXPANSION_MODE"] == "AIV"


def test_validation_rejects_unsupported_model_before_spawn(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["UnsupportedForCausalLM"],
    }))

    class Unsupported:
        SUPPORTS_TENSOR_PARALLEL = False

    monkeypatch.setattr(
        "auto_infer.models.registry.get_model_class",
        lambda _architecture: Unsupported,
    )

    with pytest.raises(ValueError, match="does not support tensor parallel"):
        validate_tp_serving(str(tmp_path))


class _Process:
    def __init__(self, *, alive=True, exitcode=None):
        self.alive = alive
        self.exitcode = exitcode
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15

    def join(self, timeout=None):
        return None

    def kill(self):
        self.killed = True
        self.alive = False
        self.exitcode = -9


def test_supervisor_terminates_every_rank_after_fatal_status():
    processes = [_Process(), _Process()]
    statuses = queue.Queue()
    statuses.put(ReplicaStatus(0, "ready"))
    statuses.put(ReplicaStatus(1, "ready"))
    statuses.put(ReplicaStatus(1, "fatal", "collective failed"))

    with pytest.raises(RuntimeError, match="rank 1.*collective failed"):
        supervise_replica(
            processes, statuses, watchdog_timeout_s=1,
            poll_interval_s=0)

    assert all(process.terminated for process in processes)


def test_supervisor_treats_one_unexpected_exit_as_replica_failure():
    processes = [_Process(alive=False, exitcode=2), _Process()]
    statuses = queue.Queue()

    with pytest.raises(RuntimeError, match="rank 0 exited"):
        supervise_replica(
            processes, statuses, watchdog_timeout_s=1,
            poll_interval_s=0)

    assert processes[1].terminated


def test_supervisor_kills_worker_that_ignores_graceful_termination():
    class StubbornProcess(_Process):
        def terminate(self):
            self.terminated = True

    processes = [StubbornProcess(), StubbornProcess()]
    statuses = queue.Queue()
    statuses.put(ReplicaStatus(1, "fatal", "collective failed"))

    with pytest.raises(RuntimeError, match="collective failed"):
        supervise_replica(
            processes, statuses, watchdog_timeout_s=1,
            poll_interval_s=0)

    assert all(process.terminated for process in processes)
    assert all(process.killed for process in processes)


def test_supervisor_watchdog_terminates_unresponsive_replica():
    processes = [_Process(), _Process()]
    statuses = queue.Queue()
    statuses.put(ReplicaStatus(0, "ready"))
    statuses.put(ReplicaStatus(1, "ready"))

    with pytest.raises(RuntimeError, match="watchdog"):
        supervise_replica(
            processes, statuses, watchdog_timeout_s=0.001,
            poll_interval_s=0.001)

    assert all(process.terminated for process in processes)


def test_supervisor_releases_replica_only_after_every_rank_is_ready():
    processes = [_Process(alive=False, exitcode=0),
                 _Process(alive=False, exitcode=0)]
    statuses = queue.Queue()
    statuses.put(ReplicaStatus(0, "ready"))
    statuses.put(ReplicaStatus(1, "ready"))
    statuses.put(ReplicaStatus(0, "stopped"))
    statuses.put(ReplicaStatus(1, "stopped"))
    ready = threading.Event()

    supervise_replica(
        processes, statuses, watchdog_timeout_s=1,
        poll_interval_s=0, ready_event=ready)

    assert ready.is_set()
