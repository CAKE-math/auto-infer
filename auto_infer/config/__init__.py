from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_path: str
    max_model_len: int = 4096
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path must not be empty")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0")


@dataclass
class ParallelConfig:
    """Parallelism degrees that DRIVE the runtime: under a multi-proc launch
    (WORLD_SIZE>1) the executor factory calls `init_distributed(self)` before
    model construction, building HCCL TP/EP/SP/multi-node groups from these
    fields. WORLD_SIZE/LOCAL_RANK come from the launcher (torchrun/deploy).
    Single card (all degrees 1 / WORLD_SIZE=1) is a no-op."""
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    sp_size: int = 1          # sequence parallel (Megatron-style, pairs with TP;
    #                           norm/residual regions sharded along sequence dim)
    nnodes: int = 1
    node_rank: int = 0

    def __post_init__(self) -> None:
        for name in ("tp_size", "ep_size", "dp_size", "cp_size", "sp_size", "nnodes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 <= self.node_rank < self.nnodes:
            raise ValueError("node_rank must satisfy 0 <= node_rank < nnodes")

    @property
    def world_size(self) -> int:
        return (self.tp_size * self.dp_size * self.ep_size *
                self.cp_size * self.sp_size)

    def to_mesh(self):
        from auto_infer.distributed.mesh import ParallelMesh
        return ParallelMesh(tp=self.tp_size, dp=self.dp_size, ep=self.ep_size,
                            cp=self.cp_size, sp=self.sp_size, nnodes=self.nnodes)


@dataclass
class CacheConfig:
    block_size: int = 16
    num_blocks: int = 1024

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    long_prefill_token_threshold: int = 0   # per-step prefill token cap (0 = disabled)

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        if self.long_prefill_token_threshold < 0:
            raise ValueError("long_prefill_token_threshold must be >= 0")


@dataclass
class SpecDecodeConfig:
    """Speculative decoding via the model's trained MTP head (greedy, KV-reuse).
    Each running request drafts 1 token per step from the MTP head (run in the
    runner off the target's hidden state, on its own paged KV); the target
    verifies it in one batched forward — greedy output stays token-identical to
    plain decode (rejection_sampler), a pure throughput win.

    MiMo ships one trained MTP layer. ``num_speculative_tokens`` controls how
    many times that layer is recurrently unrolled before target verification.
    """
    num_speculative_tokens: int = 1

    def __post_init__(self) -> None:
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be > 0")


@dataclass
class ExecutionConfig:
    """Select the runtime implementation without duplicating engine settings."""
    mode: str = "paged"  # recompute | paged | graph | graph_mtp
    device_index: int = 0
    max_gear: int = 32
    max_prefill_tokens: int = 256
    force_eager: bool = False

    def __post_init__(self) -> None:
        from auto_infer.executor_backends import has_executor_backend
        if not has_executor_backend(self.mode):
            raise ValueError(f"unsupported execution mode: {self.mode}")
        if self.device_index < 0:
            raise ValueError("device_index must be >= 0")
        if self.max_gear <= 0:
            raise ValueError("max_gear must be > 0")
        if self.max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be > 0")


@dataclass
class EngineConfig:
    model: ModelConfig
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    # Zero-host-bubble graph decode is opt-in because its current supported
    # boundary is history-independent greedy and prefill uses a safe eager
    # barrier. The July-23 Qwen3 BF16 gate passed decode TPOT/throughput.
    async_scheduling: bool = False
    async_batches: int = 2       # in-flight batch-queue depth (vLLM max_concurrent_batches)
    log_stats: bool = False           # periodic StatLogger metrics line (off by default)
    log_stats_interval_s: float = 5.0  # seconds between metrics lines
    spec_decode: "SpecDecodeConfig | None" = None   # None = disabled (plain 1-token decode)

    def __post_init__(self) -> None:
        if self.async_batches <= 0:
            raise ValueError("async_batches must be > 0")
        if self.log_stats_interval_s <= 0:
            raise ValueError("log_stats_interval_s must be > 0")
        if self.spec_decode is not None and self.async_scheduling:
            raise ValueError(
                "speculative decoding cannot be combined with async_scheduling; "
                "the MTP executor already owns its device pipeline")
