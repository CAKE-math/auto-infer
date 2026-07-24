"""Model-owned tensor-parallel weight layout contracts."""

from dataclasses import dataclass


SliceSpec = tuple[int, int, int]


@dataclass(frozen=True)
class TensorParallelPlan:
    """One Qwen rank's checkpoint slicing plan before projection packing."""

    rank: int
    size: int
    q_rows: int
    kv_rows: int
    intermediate_rows: int

    @classmethod
    def for_qwen(cls, config, rank: int, size: int) -> "TensorParallelPlan":
        if size <= 0:
            raise ValueError("tensor-parallel size must be > 0")
        if not 0 <= rank < size:
            raise ValueError(
                "tensor-parallel rank must satisfy 0 <= rank < size")
        for field in ("num_heads", "num_kv_heads", "intermediate_size"):
            if getattr(config, field) % size:
                raise ValueError(f"{field} must be divisible by tp_size")
        return cls(
            rank=rank,
            size=size,
            q_rows=config.num_heads // size * config.head_dim,
            kv_rows=config.num_kv_heads // size * config.head_dim,
            intermediate_rows=config.intermediate_size // size,
        )

    def slice_spec(self, name: str) -> "SliceSpec | None":
        if not name.startswith("model.layers."):
            return None
        suffixes = {
            "self_attn.q_proj.weight": (0, self.q_rows),
            "self_attn.q_proj.bias": (0, self.q_rows),
            "self_attn.k_proj.weight": (0, self.kv_rows),
            "self_attn.k_proj.bias": (0, self.kv_rows),
            "self_attn.v_proj.weight": (0, self.kv_rows),
            "self_attn.v_proj.bias": (0, self.kv_rows),
            "self_attn.o_proj.weight": (1, self.q_rows),
            "mlp.gate_proj.weight": (0, self.intermediate_rows),
            "mlp.up_proj.weight": (0, self.intermediate_rows),
            "mlp.down_proj.weight": (1, self.intermediate_rows),
        }
        for suffix, (dimension, length) in suffixes.items():
            if name.endswith(suffix):
                return dimension, self.rank * length, length
        return None
