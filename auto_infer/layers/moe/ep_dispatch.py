"""Ascend fused expert-parallel token dispatch/combine protocol."""
from dataclasses import dataclass
from importlib import import_module

import torch

from auto_infer.distributed.topology import ExpertParallelTopology


@dataclass(frozen=True)
class DispatchResult:
    """Opaque dispatch metadata plus expert-sorted local activations."""

    hidden_states: torch.Tensor
    dynamic_scale: torch.Tensor | None
    expand_idx: torch.Tensor
    expert_tokens: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor


@dataclass(frozen=True)
class MoeDispatchQuantization:
    """Stable input policy seam; only unquantized BF16 is enabled today."""

    quant_mode: int = 0
    scales: torch.Tensor | None = None


class NpuMoeDispatchCombine:
    """BF16-only wrapper around CANN's fused MoE EP collectives."""

    def __init__(self, topology: ExpertParallelTopology, num_experts: int,
                 dtype: torch.dtype, ops=None,
                 quantization: MoeDispatchQuantization | None = None):
        if dtype is not torch.bfloat16:
            raise NotImplementedError(
                "fused EP dispatch currently supports BF16 only")
        if topology.world_size <= 1:
            raise ValueError("fused EP dispatch requires EP world size > 1")
        if not 0 <= topology.rank < topology.world_size:
            raise ValueError("EP rank is outside the configured world size")
        if topology.group is None:
            raise ValueError("EP process group is unavailable")
        if num_experts % topology.world_size:
            raise ValueError("global expert count must be divisible by EP size")
        if not topology.hccl_comm_name:
            raise ValueError("EP HCCL communicator name is unavailable")
        quantization = quantization or MoeDispatchQuantization()
        if quantization.quant_mode != 0 or quantization.scales is not None:
            raise NotImplementedError(
                "quantized EP dispatch is not enabled; BF16 mode 0 is required")
        ops = import_module("torch_npu") if ops is None else ops
        for name in (
            "npu_moe_distribute_dispatch_v2",
            "npu_moe_distribute_combine_v2",
        ):
            if not callable(getattr(ops, name, None)):
                raise RuntimeError(f"torch-npu is missing {name}")
        self.topology = topology
        self.num_experts = num_experts
        self.dtype = dtype
        self.ops = ops
        self.quantization = quantization
        self.dispatch_calls = 0
        self.combine_calls = 0

    @staticmethod
    def _validate_routing(token_count: int, expert_ids: torch.Tensor,
                          active_token_mask: torch.Tensor | None) -> None:
        if expert_ids.ndim != 2 or expert_ids.shape[0] != token_count:
            raise ValueError("expert IDs must have shape (tokens, top_k)")
        if expert_ids.dtype is not torch.int32:
            raise ValueError("expert IDs must use int32")
        if active_token_mask is None:
            return
        if (active_token_mask.dtype is not torch.bool
                or active_token_mask.ndim != 1
                or active_token_mask.shape[0] != token_count):
            raise ValueError(
                "active-token mask must be a token-shaped bool tensor")

    def dispatch(self, x: torch.Tensor, expert_ids: torch.Tensor,
                 active_token_mask: torch.Tensor | None = None) -> DispatchResult:
        if x.ndim != 2:
            raise ValueError("EP dispatch input must have shape (tokens, hidden)")
        if x.dtype is not torch.bfloat16:
            raise ValueError("EP dispatch input must use BF16")
        self._validate_routing(x.shape[0], expert_ids, active_token_mask)
        output = self.ops.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            scales=self.quantization.scales,
            quant_mode=self.quantization.quant_mode,
            group_ep=self.topology.hccl_comm_name,
            ep_world_size=self.topology.world_size,
            ep_rank_id=self.topology.rank,
            x_active_mask=active_token_mask,
        )
        if not isinstance(output, (tuple, list)) or len(output) < 6:
            raise RuntimeError("dispatch_v2 returned an invalid metadata tuple")
        hidden, dynamic_scale, expand_idx, expert_tokens, ep_counts, tp_counts = output[:6]
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
                or hidden.dtype is not self.dtype
                or hidden.shape[1] != x.shape[1]):
            raise RuntimeError("dispatch_v2 must return a hidden-width BF16 matrix")
        if (not isinstance(expert_tokens, torch.Tensor)
                or expert_tokens.ndim != 1):
            raise RuntimeError("dispatch_v2 must return one-dimensional expert counts")
        for name, value in (
            ("expand indices", expand_idx),
            ("EP receive counts", ep_counts),
            ("TP receive counts", tp_counts),
        ):
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"dispatch_v2 returned invalid {name}")
        self.dispatch_calls += 1
        return DispatchResult(
            hidden_states=hidden,
            dynamic_scale=dynamic_scale,
            expand_idx=expand_idx,
            expert_tokens=expert_tokens.to(torch.int64),
            ep_recv_counts=ep_counts,
            tp_recv_counts=tp_counts,
        )

    def combine(self, hidden_states: torch.Tensor, expert_ids: torch.Tensor,
                expert_weights: torch.Tensor, metadata: DispatchResult,
                active_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_routing(
            expert_ids.shape[0], expert_ids, active_token_mask)
        if hidden_states.ndim != 2 or hidden_states.dtype is not self.dtype:
            raise ValueError("local expert output must be a two-dimensional BF16 tensor")
        if expert_weights.shape != expert_ids.shape:
            raise ValueError("expert weights must match expert IDs")
        output = self.ops.npu_moe_distribute_combine_v2(
            expand_x=hidden_states,
            expert_ids=expert_ids,
            assist_info_for_combine=metadata.expand_idx,
            expert_scales=expert_weights.to(torch.float32),
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_experts,
            global_bs=0,
            ep_send_counts=metadata.ep_recv_counts,
            group_ep=self.topology.hccl_comm_name,
            ep_world_size=self.topology.world_size,
            ep_rank_id=self.topology.rank,
            tp_send_counts=metadata.tp_recv_counts,
            x_active_mask=active_token_mask,
        )
        expected_shape = (expert_ids.shape[0], hidden_states.shape[1])
        if (not isinstance(output, torch.Tensor)
                or output.dtype is not self.dtype
                or tuple(output.shape) != expected_shape):
            raise RuntimeError(
                "combine_v2 must return a BF16 matrix with source-token shape")
        self.combine_calls += 1
        return output
