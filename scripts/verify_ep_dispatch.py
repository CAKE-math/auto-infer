"""BF16 Moonlight EP dispatch/combine parity and graph acceptance on Ascend."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def summarize(rank_results: list[dict]) -> dict:
    """Reduce per-rank acceptance evidence without importing torch-npu."""
    max_abs_error = max(result["max_abs_error"] for result in rank_results)
    allclose = all(result["allclose"] for result in rank_results)
    token_identity = all(result["token_identity"] for result in rank_results)
    collectives = all(
        result["dispatch_calls"] > 0
        and result["dispatch_calls"] == result["combine_calls"]
        for result in rank_results)
    routed_all_reduce = any(
        result["routed_all_reduce_calls"] > 0 for result in rank_results)
    return {
        "max_abs_error": max_abs_error,
        "allclose": allclose,
        "token_identity": token_identity,
        "dispatch_combine_observed": collectives,
        "routed_all_reduce_observed": routed_all_reduce,
        "passed": (allclose and token_identity and collectives
                   and not routed_all_reduce),
    }


def benchmark_summary(all_to_all_s: list[float],
                      all_reduce_s: list[float]) -> dict:
    """Summarize paired, steady-state layer timings."""
    all_to_all = statistics.median(all_to_all_s)
    all_reduce = statistics.median(all_reduce_s)
    return {
        "all_to_all_median_ms": all_to_all * 1000,
        "all_reduce_median_ms": all_reduce * 1000,
        "all_to_all_speedup": all_reduce / all_to_all,
    }


def _logit_parity(all_to_all, all_reduce) -> dict:
    """Describe last-token logit drift and whether it crosses greedy argmax."""
    import torch

    new = all_to_all[-1].float()
    old = all_reduce[-1].float()
    delta = (new - old).abs()
    new_top = new.topk(2)
    old_top = old.topk(2)
    return {
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            new.unsqueeze(0), old.unsqueeze(0)).item()),
        "old_argmax": int(old_top.indices[0].item()),
        "new_argmax": int(new_top.indices[0].item()),
        "argmax_identity": bool(
            old_top.indices[0].item() == new_top.indices[0].item()),
        "old_top1_margin": float(
            (old_top.values[0] - old_top.values[1]).item()),
        "new_top1_margin": float(
            (new_top.values[0] - new_top.values[1]).item()),
    }


def _all_reduce_expert_reference(
        x, topk_ids, topk_weights, w13_local, w2_local, lo, n_local):
    """Numerical reference for the retired routed-output all-reduce path."""
    import torch

    from auto_infer.layers.moe.fused_moe import fused_experts

    hidden, inter2 = w13_local.shape[1], w13_local.shape[2]
    inter = w2_local.shape[1]
    zero13 = torch.zeros(
        1, hidden, inter2, dtype=w13_local.dtype, device=w13_local.device)
    zero2 = torch.zeros(
        1, inter, hidden, dtype=w2_local.dtype, device=w2_local.device)
    w13 = torch.cat([w13_local, zero13], dim=0)
    w2 = torch.cat([w2_local, zero2], dim=0)
    local = (topk_ids >= lo) & (topk_ids < lo + n_local)
    ids = torch.where(
        local, topk_ids - lo, torch.full_like(topk_ids, n_local))
    return fused_experts(x, ids, topk_weights, w13, w2, n_local + 1)


def _ep_all_reduce(tensor):
    """Verification-only reference collective; not part of runtime EP."""
    import torch.distributed as dist

    from auto_infer.distributed.parallel_state import ep_size, ep_topology

    if ep_size() > 1:
        dist.all_reduce(tensor, group=ep_topology().group)
    return tensor


def _install_all_reduce_reference(model) -> dict[str, int]:
    """Install the retired EP path on one model for end-to-end A/B testing."""
    from types import MethodType

    import torch

    from auto_infer.distributed.parallel_state import ep_rank, ep_size
    from auto_infer.layers.mlp import swiglu_mlp
    from auto_infer.layers.moe.fused_moe import build_expert_weights

    calls = {"calls": 0}

    def all_reduce_ep(moe, x, layer, active_token_mask=None):
        del active_token_mask
        cfg, weights = model.cfg, model.w
        prefix = moe.layer_prefix(layer) + "mlp."
        local_experts = cfg.n_routed // ep_size()
        first_expert = ep_rank() * local_experts
        if layer not in moe._fused_w_ep:
            moe._fused_w_ep[layer] = build_expert_weights(
                weights, prefix, cfg.n_routed,
                first_expert, first_expert + local_experts)
            if moe.free_originals:
                for expert in range(first_expert,
                                    first_expert + local_experts):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        weights.pop(
                            f"{prefix}experts.{expert}.{projection}.weight",
                            None)
        w13, w2 = moe._fused_w_ep[layer]
        router = (x @ weights[prefix + "gate.weight"].t()).float()
        topk_weights, topk_ids = moe._gate(router, prefix)
        topk_weights = (topk_weights * cfg.routed_scale).to(model.dtype)
        routed = _all_reduce_expert_reference(
            x, topk_ids.to(torch.int32), topk_weights, w13, w2,
            first_expert, local_experts)
        calls["calls"] += 1
        routed = _ep_all_reduce(routed)
        return routed + swiglu_mlp(
            x, weights, prefix + "shared_experts.")

    model.moe._fused_ep = MethodType(all_reduce_ep, model.moe)
    return calls


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep-size", required=True, type=int)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument(
        "--ep-backend", choices=("all-to-all", "all-reduce"),
        default="all-to-all")
    parser.add_argument("--prompt-token-ids", default="100,200,300,400")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-gear", type=int, default=16)
    parser.add_argument("--output-dir", default="results/ep-dispatch")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reference-json")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument("--operator-only", action="store_true")
    parser.add_argument("--model-parity-only", action="store_true")
    parser.add_argument("--benchmark-layer", action="store_true")
    parser.add_argument("--benchmark-warmups", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--layer-token-count", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _layer_parity(model, token_count: int, args):
    import torch

    from auto_infer.distributed.parallel_state import ep_rank, ep_size
    from auto_infer.layers.mlp import swiglu_mlp

    cfg = model.cfg
    layer = cfg.first_k_dense
    prefix = model.layer_prefix(layer) + "mlp."
    values = torch.arange(
        token_count * cfg.hidden_size, dtype=torch.int32,
        device=model.device).reshape(token_count, cfg.hidden_size)
    x = ((values.remainder(31).float() - 15) / 32).to(torch.bfloat16)

    def all_to_all_path():
        return model.moe._fused_ep(x, layer)

    routed_and_shared = all_to_all_path()
    adapter = model.moe._ep_dispatch
    w13, w2 = model.moe._fused_w_ep[layer]
    local_experts = cfg.n_routed // ep_size()
    first_expert = ep_rank() * local_experts

    def all_reduce_path():
        router = (x @ model.w[prefix + "gate.weight"].t()).float()
        topk_weights, topk_ids = model.moe._gate(router, prefix)
        topk_weights = (topk_weights * cfg.routed_scale).to(model.dtype)
        topk_ids = topk_ids.to(torch.int32)
        routed = _ep_all_reduce(_all_reduce_expert_reference(
            x, topk_ids, topk_weights, w13, w2,
            first_expert, local_experts))
        return routed + swiglu_mlp(
            x, model.w, prefix + "shared_experts.")

    reference = all_reduce_path()
    torch.npu.synchronize()
    delta = (routed_and_shared.float() - reference.float()).abs()
    result = {
        "max_abs_error": float(delta.max().item()),
        "allclose": bool(torch.allclose(
            routed_and_shared.float(), reference.float(),
            atol=5e-2, rtol=1e-2)),
        "adapter": adapter,
    }
    if not args.benchmark_layer:
        return result

    def elapsed(fn):
        torch.npu.synchronize()
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        return time.perf_counter() - started

    for _ in range(args.benchmark_warmups):
        all_to_all_path()
        all_reduce_path()
    torch.npu.synchronize()
    all_to_all_s, all_reduce_s = [], []
    for iteration in range(args.benchmark_repeats):
        if iteration % 2:
            all_reduce_s.append(elapsed(all_reduce_path))
            all_to_all_s.append(elapsed(all_to_all_path))
        else:
            all_to_all_s.append(elapsed(all_to_all_path))
            all_reduce_s.append(elapsed(all_reduce_path))
    result["layer_benchmark"] = {
        **benchmark_summary(all_to_all_s, all_reduce_s),
        "token_count": token_count,
        "warmups": args.benchmark_warmups,
        "repeats": args.benchmark_repeats,
        "all_to_all_samples_ms": [value * 1000 for value in all_to_all_s],
        "all_reduce_samples_ms": [value * 1000 for value in all_reduce_s],
    }
    return result


def _operator_identity_parity(args):
    """Exercise dispatch/combine with identity experts at a valid A2 shape."""
    import torch

    from auto_infer.distributed.parallel_state import ep_topology
    from auto_infer.layers.moe.ep_dispatch import NpuMoeDispatchCombine

    token_count = 8
    device = torch.device(f"npu:{int(os.environ.get('LOCAL_RANK', '0'))}")
    values = torch.arange(
        token_count * args.hidden_size, dtype=torch.int32, device=device)
    x = ((values.reshape(token_count, args.hidden_size).remainder(31).float()
          - 15) / 32).to(torch.bfloat16)
    expert_ids = (torch.arange(
        token_count * args.top_k, dtype=torch.int32, device=device)
        .reshape(token_count, args.top_k).remainder(args.num_experts))
    weights = torch.full(
        (token_count, args.top_k), 1.0 / args.top_k,
        dtype=torch.bfloat16, device=device)
    adapter = NpuMoeDispatchCombine(
        ep_topology(), args.num_experts, torch.bfloat16)
    metadata = adapter.dispatch(x, expert_ids)
    output = adapter.combine(
        metadata.hidden_states, expert_ids, weights, metadata)
    torch.npu.synchronize()
    delta = (output.float() - x.float()).abs()
    return {
        "max_abs_error": float(delta.max().item()),
        "allclose": bool(torch.allclose(
            output.float(), x.float(), atol=5e-2, rtol=1e-2)),
        "adapter": adapter,
    }


def _prompt_token_ids(model, args) -> list[int]:
    with open(Path(args.model) / "config.json") as config_file:
        model_config = json.load(config_file)
    prompt = [int(token) for token in args.prompt_token_ids.split(",")]
    bos_token_id = model_config.get("bos_token_id")
    if bos_token_id is not None:
        prompt.insert(0, int(bos_token_id))
    if not prompt or any(token < 0 or token >= model.cfg.vocab_size
                         for token in prompt):
        raise ValueError("prompt token IDs must be inside the model vocabulary")
    return prompt


def _model_parity(model, args):
    """Compare full-model prefill logits before autoregressive divergence."""
    import torch

    prompt = _prompt_token_ids(model, args)
    token_ids = torch.tensor(prompt, dtype=torch.long, device=model.device)
    positions = torch.arange(
        len(prompt), dtype=torch.long, device=model.device)
    all_to_all_ep = model.moe._fused_ep
    all_to_all_logits = model.forward_dense(token_ids, positions)
    reference_calls = _install_all_reduce_reference(model)
    all_reduce_logits = model.forward_dense(token_ids, positions)
    model.moe._fused_ep = all_to_all_ep
    torch.npu.synchronize()
    return {
        **_logit_parity(all_to_all_logits, all_reduce_logits),
        "reference_moe_calls": reference_calls["calls"],
    }


def _build_executor(model, args):
    from auto_infer.engine.executor import RunnerExecutor
    if args.mode == "graph":
        from auto_infer.worker.graph_decode_runner import GraphPagedRunner
        runner = GraphPagedRunner(
            model, num_blocks=512, block_size=16,
            max_gear=args.max_gear, max_model_len=4096)
    else:
        from auto_infer.worker.model_runner import NpuModelRunner
        runner = NpuModelRunner(
            model, num_blocks=512, block_size=16,
            max_num_batched_tokens=2048, max_num_seqs=args.max_gear,
            max_model_len=4096)
    return RunnerExecutor(runner)


def _generate(model, args):
    import torch

    from auto_infer.config import (
        CacheConfig, EngineConfig, ExecutionConfig, ModelConfig,
        ParallelConfig, SchedulerConfig)
    from auto_infer.entrypoints.llm import LLM
    from auto_infer.layers import sampler as sampler_module

    prompt = _prompt_token_ids(model, args)
    config = EngineConfig(
        model=ModelConfig(args.model, max_model_len=4096, dtype="bfloat16"),
        parallel=ParallelConfig(ep_size=args.ep_size),
        cache=CacheConfig(block_size=16, num_blocks=512),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_gear, max_num_batched_tokens=2048),
        execution=ExecutionConfig(
            mode="graph" if args.mode == "graph" else "paged",
            device_index=int(os.environ.get("LOCAL_RANK", "0")),
            max_gear=args.max_gear),
    )
    first_logits = {}
    sample_batched = sampler_module.sample_batched

    def traced_sample_batched(logits, sampling):
        if not first_logits:
            row = logits[0].float()
            top = row.topk(2)
            first_logits.update({
                "argmax": int(top.indices[0].item()),
                "top2": int(top.indices[1].item()),
                "top1_logit": float(top.values[0].item()),
                "top2_logit": float(top.values[1].item()),
                "top1_margin": float(
                    (top.values[0] - top.values[1]).item()),
            })
        return sample_batched(logits, sampling)

    sampler_module.sample_batched = traced_sample_batched
    executor = None
    samples = []
    tokens = None
    try:
        executor = _build_executor(model, args)
        for iteration in range(args.warmups + args.repeats):
            llm = LLM(config, executor=executor)
            torch.npu.synchronize()
            started = time.perf_counter()
            tokens = llm.generate(
                [prompt] * args.batch_size, max_tokens=args.max_tokens)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            if iteration >= args.warmups:
                samples.append(elapsed)
    finally:
        if executor is not None:
            executor.close()
        sampler_module.sample_batched = sample_batched
    generated = args.batch_size * args.max_tokens
    return tokens, samples, generated / min(samples), first_logits


def _reference_tokens(path: str | None):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return payload["tokens"]


def main():
    args = _parse_args()
    if args.ep_size < 1:
        raise ValueError("ep-size must be positive")
    if (args.layer_token_count < 1 or args.benchmark_warmups < 0
            or args.benchmark_repeats < 1):
        raise ValueError("layer benchmark counts are invalid")
    if (args.ep_size > 1 and not args.reference_json and not args.reference_only
            and not args.layer_only and not args.operator_only
            and not args.model_parity_only):
        raise ValueError("EP acceptance requires --reference-json from EP1")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from auto_infer.config import ParallelConfig
    from auto_infer.distributed import parallel_state as ps
    from auto_infer.engine.factory import load_model

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"rank {local_rank}: initialize device/distributed", flush=True)
    torch.npu.set_device(local_rank)
    ps.init_distributed(ParallelConfig(ep_size=args.ep_size))
    if args.operator_only:
        print(f"rank {local_rank}: run identity operator parity", flush=True)
        parity = _operator_identity_parity(args)
        adapter = parity.pop("adapter")
        model = None
    else:
        print(f"rank {local_rank}: load model shard", flush=True)
        model = load_model(args.model, local_rank, "bfloat16")
        if args.ep_size > 1:
            print(f"rank {local_rank}: run layer parity", flush=True)
            parity = _layer_parity(
                model, token_count=args.layer_token_count, args=args)
            adapter = parity.pop("adapter")
        else:
            parity = {"max_abs_error": 0.0, "allclose": True}
            adapter = None

    routed_all_reduce = {"calls": 0}
    acceptance_only = (
        args.layer_only or args.operator_only or args.model_parity_only)
    dispatch_calls_before = (
        0 if acceptance_only or adapter is None else adapter.dispatch_calls)
    combine_calls_before = (
        0 if acceptance_only or adapter is None else adapter.combine_calls)
    if (model is not None and args.ep_size > 1
            and args.ep_backend == "all-reduce"):
        routed_all_reduce = _install_all_reduce_reference(model)

    if args.model_parity_only:
        parity["model_parity"] = _model_parity(model, args)
        tokens, elapsed_samples, throughput = [], [], 0.0
        first_logits = None
        token_identity = parity["model_parity"]["argmax_identity"]
    elif args.layer_only or args.operator_only:
        tokens, elapsed_samples, throughput = [], [], 0.0
        first_logits = None
        token_identity = True
    else:
        print(f"rank {local_rank}: run {args.mode} generation", flush=True)
        tokens, elapsed_samples, throughput, first_logits = _generate(
            model, args)
        print(f"rank {local_rank}: generation complete", flush=True)
        expected = _reference_tokens(args.reference_json)
        token_identity = expected is None or tokens == expected
    result = {
        "rank": ps.ep_rank(),
        "ep_size": ps.ep_size(),
        "mode": args.mode,
        "ep_backend": args.ep_backend,
        "max_abs_error": parity["max_abs_error"],
        "allclose": parity["allclose"],
        "token_identity": token_identity,
        "dispatch_calls": (
            adapter.dispatch_calls - dispatch_calls_before if adapter else 0),
        "combine_calls": (
            adapter.combine_calls - combine_calls_before if adapter else 0),
        "routed_all_reduce_calls": routed_all_reduce["calls"],
        "elapsed_samples_s": elapsed_samples,
        "best_throughput_tok_s": throughput,
        "first_logits": first_logits,
        "layer_benchmark": parity.get("layer_benchmark"),
        "model_parity": parity.get("model_parity"),
        "tokens": tokens,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"ep{args.ep_size}-{args.mode}"
    rank_path = output_dir / f"{run_id}-rank{ps.ep_rank()}.json"
    rank_path.write_text(json.dumps(result, indent=2))
    if args.ep_size > 1:
        dist.barrier(group=ps._ep_group())

    if ps.ep_rank() == 0:
        if args.ep_size == 1:
            summary = {
                "reference_only": True,
                "tokens": tokens,
                "elapsed_samples_s": elapsed_samples,
                "best_throughput_tok_s": throughput,
            }
        else:
            ranks = [json.loads(
                (output_dir / f"{run_id}-rank{rank}.json").read_text())
                for rank in range(args.ep_size)]
            summary = {**summarize(ranks), "ranks": ranks, "tokens": tokens}
        summary_path = output_dir / f"{run_id}-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if not args.reference_only and args.ep_size > 1 and not summary["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
