from dataclasses import dataclass
from math import prod

from auto_infer.errors import ConfigurationError


_AXES = ("dp", "cp", "sp", "ep", "tp")


@dataclass(frozen=True)
class ParallelMesh:
    tp: int = 1
    dp: int = 1
    ep: int = 1
    cp: int = 1
    sp: int = 1
    nnodes: int = 1

    def __post_init__(self):
        for axis in _AXES:
            if getattr(self, axis) <= 0:
                raise ConfigurationError(f"{axis} must be > 0")
        if self.nnodes <= 0:
            raise ConfigurationError("nnodes must be > 0")

    @property
    def world_size(self) -> int:
        return prod(getattr(self, axis) for axis in _AXES)

    def validate(self, world_size: int) -> None:
        if world_size != self.world_size:
            raise ConfigurationError(
                f"WORLD_SIZE={world_size} does not match parallel mesh; "
                f"mesh requires {self.world_size}")
        if world_size % self.nnodes:
            raise ConfigurationError(
                f"WORLD_SIZE={world_size} is not divisible by nnodes={self.nnodes}")

    def coordinate(self, rank: int) -> tuple[int, ...]:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} outside [0, {self.world_size})")
        coords = []
        remaining = rank
        for axis in reversed(_AXES):
            size = getattr(self, axis)
            coords.append(remaining % size)
            remaining //= size
        return tuple(reversed(coords))

    def groups(self, axis: str) -> list[list[int]]:
        if axis not in _AXES:
            raise ValueError(f"unknown parallel axis: {axis}")
        index = _AXES.index(axis)
        groups = {}
        for rank in range(self.world_size):
            coord = self.coordinate(rank)
            key = coord[:index] + coord[index + 1:]
            groups.setdefault(key, []).append(rank)
        return list(groups.values())
