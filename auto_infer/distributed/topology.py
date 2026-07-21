"""Immutable distributed resource contracts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertParallelTopology:
    """Runtime resources required by fused expert-parallel collectives."""

    group: object
    rank: int
    world_size: int
    hccl_comm_name: str | None
