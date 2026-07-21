"""Real NPU executor: replaces MockExecutor with a true model forward on Ascend.

Implements the same Executor interface EngineCore depends on, so wiring this in
changes nothing in engine/scheduler/kv. Milestone 1: recompute full sequence per
step (no KV reuse); paged-KV + FIA attention is the next swap.
"""
import torch

from auto_infer.engine.executor import Executor
from auto_infer.engine.execution import BatchPlan, ExecutionResult


class NpuExecutor(Executor):
    """Model-agnostic NPU executor (spec sec 7): picks the model class from
    config.architectures via the registry. Any registered model with
    .from_pretrained(path, device, dtype) + .forward(token_ids, positions)
    plugs in unchanged — no engine/scheduler change."""

    def __init__(self, model_path: str, device_index: int = 0, dtype: str = "bfloat16"):
        from auto_infer.engine.factory import load_model
        self.model = load_model(model_path, device_index, dtype)
        self.device = self.model.device

    def execute(self, plan: BatchPlan) -> ExecutionResult:
        sampled: dict[str, int] = {}
        for sr in plan.scheduled:
            req = plan.get_request(sr.request_id)
            computed_after = req.num_computed_tokens + sr.num_tokens_to_compute
            if computed_after >= req.num_prefill_tokens:
                ids = req.all_token_ids
                t = torch.tensor(ids, dtype=torch.long, device=self.device)
                pos = torch.arange(len(ids), dtype=torch.long, device=self.device)
                fwd = getattr(self.model, "forward_dense", self.model.forward)
                logits = fwd(t, pos)
                sampled[req.request_id] = int(logits[-1].float().argmax().item())
        return ExecutionResult.from_single_tokens(sampled)
