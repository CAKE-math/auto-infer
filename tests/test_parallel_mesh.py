import sys
from types import SimpleNamespace

import pytest
import torch

from auto_infer.config import ParallelConfig
from auto_infer.distributed import parallel_state as ps
from auto_infer.distributed.mesh import ParallelMesh
from auto_infer.errors import ConfigurationError


def test_named_axis_groups_cover_world_once_per_axis():
    mesh = ParallelMesh(tp=2, dp=2, ep=2, cp=1, sp=2)
    assert mesh.world_size == 16
    for axis in ("tp", "dp", "ep", "cp", "sp"):
        groups = mesh.groups(axis)
        assert sorted(rank for group in groups for rank in group) == list(range(16))
        assert all(len(group) == getattr(mesh, axis) for group in groups)


def test_coordinates_are_bijective():
    mesh = ParallelMesh(tp=2, dp=2, ep=2, cp=1, sp=2)
    assert len({mesh.coordinate(rank) for rank in range(mesh.world_size)}) == mesh.world_size


def test_world_size_mismatch_is_rejected():
    with pytest.raises(ConfigurationError, match=r"WORLD_SIZE=2.*mesh requires 4"):
        ParallelMesh(tp=2, dp=2).validate(2)


def test_invalid_axis_size_is_rejected():
    with pytest.raises(ConfigurationError, match="tp must be > 0"):
        ParallelMesh(tp=0)


def test_init_distributed_skips_singleton_axis_groups(monkeypatch):
    created = []

    class _Backend:
        def get_hccl_comm_name(self, rank):
            return f"ep-{rank}"

    class _Group:
        def _get_backend(self, device):
            return _Backend()

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(set_device=lambda rank: None),
        raising=False)
    monkeypatch.setattr(ps.torch, "device", lambda name: name)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "new_group",
        lambda members: created.append(tuple(members)) or _Group())
    monkeypatch.setattr(ps, "_INITED", False)

    ps.init_distributed(ParallelConfig(ep_size=4))

    assert created == [(0, 1, 2, 3)]
    assert ps.ep_size() == 4
    assert ps.ep_rank() == 0
    assert ps.ep_topology().hccl_comm_name == "ep-0"


def test_init_distributed_rejects_world_mismatch_before_npu_setup(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(ConfigurationError, match="WORLD_SIZE=1"):
        ps.init_distributed(ParallelConfig(ep_size=2))


def test_sp_rank_does_not_fall_back_to_tp_rank(monkeypatch):
    monkeypatch.setattr(ps, "_SP_RANK", 0)
    monkeypatch.setattr(ps, "_SP_GROUP", None)
    monkeypatch.setattr(ps, "_TP_RANK", 3)

    assert ps.sp_rank() == 0


def test_reinitialization_rejects_a_different_parallel_mesh(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(ps, "_INITED", True)
    monkeypatch.setattr(ps, "_TP_SIZE", 2)
    monkeypatch.setattr(ps, "_DP_SIZE", 1)
    monkeypatch.setattr(ps, "_EP_SIZE", 2)
    monkeypatch.setattr(ps, "_CP_SIZE", 1)
    monkeypatch.setattr(ps, "_SP_SIZE", 1)

    with pytest.raises(ConfigurationError, match="already initialized"):
        ps.init_distributed(ParallelConfig(ep_size=4))
