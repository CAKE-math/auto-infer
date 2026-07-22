"""Capture one bounded, matched Qwen3 request as a Chrome Trace artifact."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import load_manifest, token_digest
from benchmarks.qwen3_profile_common import (
    sha256_file,
    validate_chrome_trace,
    write_profile_metadata,
)


SUPPORTED_FRAMEWORKS = {"auto-infer", "omni-npu", "vllm-ascend"}
PROFILE_OUTPUT_TOKENS = 16
PROFILE_WARMUP_OUTPUT_TOKENS = 8


def profile_configuration(manifest: dict, framework: str) -> dict:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"unsupported framework: {framework}")
    return {
        "framework": framework,
        "model": manifest["model"],
        "prompt": manifest["prompt"],
        "batch_size": manifest["throughput_batch"],
        "output_tokens": PROFILE_OUTPUT_TOKENS,
        "warmup_runs": manifest["warmup_runs"],
        "warmup_output_tokens": PROFILE_WARMUP_OUTPUT_TOKENS,
        "max_model_len": manifest["max_model_len"],
        "usable_kv_tokens": manifest["usable_kv_tokens"],
        "kv_block_size": manifest["kv_block_size"],
        "kv_cache_memory_bytes": manifest["kv_cache_memory_bytes"],
        "dtype": manifest["dtype"],
        "temperature": manifest["temperature"],
        "seed": manifest["seed"],
        "capture_phases": {
            "prefill_passes": 1,
            "decode_passes": max(PROFILE_OUTPUT_TOKENS - 1, 0),
            "continuous_decode": PROFILE_OUTPUT_TOKENS > 1,
            "speculative_mtp": False,
        },
    }


def prepare_omni_compatibility(model: str) -> None:
    """Supply the legacy argv model slot read by omni-npu during import."""
    while len(sys.argv) <= 2:
        sys.argv.append("")
    sys.argv[2] = model


def _revision() -> str:
    explicit = os.environ.get("AUTO_INFER_SOURCE_REVISION")
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True,
        text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _driver_version() -> dict:
    path = Path("/usr/local/Ascend/driver/version.info")
    if not path.is_file():
        return {}
    entries = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {"Version", "ascendhal_version", "package_version"}:
                entries[key] = value
    return entries


def _model_identity(model: str, prompt_ids: list[int]) -> dict:
    model_path = Path(model)
    config_path = model_path / "config.json"
    checkpoints = sorted(model_path.glob("*.safetensors"))
    if not config_path.is_file() or not checkpoints:
        raise ValueError("profiling requires local config and safetensors files")
    if len(checkpoints) == 1:
        checkpoint_digest = sha256_file(checkpoints[0])
    else:
        combined = hashlib.sha256()
        for checkpoint in checkpoints:
            combined.update(checkpoint.name.encode())
            combined.update(sha256_file(checkpoint).encode())
        checkpoint_digest = combined.hexdigest()
    return {
        "path": model,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_digest,
        "prompt_token_ids": list(prompt_ids),
        "prompt_token_count": len(prompt_ids),
    }


def _environment(torch, torch_npu) -> dict:
    device_name = "unknown"
    get_name = getattr(torch.npu, "get_device_name", None)
    if get_name:
        device_name = get_name(0)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device_name,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "vllm": _package_version("vllm"),
        "omni_plugin": os.environ.get("VLLM_PLUGINS", ""),
        "visible_npus": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "driver": _driver_version(),
        "capture_harness_revision": _revision(),
        "capture_harness_revision_origin": "capture_environment",
        "framework_source_revisions": json.loads(os.environ.get(
            "PROFILE_FRAMEWORK_SOURCE_REVISIONS", "{}")),
    }


class _AutoInferAdapter:
    def __init__(self, manifest: dict):
        from transformers import AutoTokenizer

        from auto_infer.config import (
            CacheConfig, EngineConfig, ExecutionConfig, ModelConfig,
            SchedulerConfig)
        from auto_infer.entrypoints.llm import LLM

        block_size = manifest["kv_block_size"]
        if manifest["usable_kv_tokens"] % block_size:
            raise ValueError("usable_kv_tokens must be divisible by kv_block_size")
        self._prompt_ids = AutoTokenizer.from_pretrained(
            manifest["model"], trust_remote_code=True)(
                manifest["prompt"]).input_ids
        config = EngineConfig(
            model=ModelConfig(
                manifest["model"], max_model_len=manifest["max_model_len"],
                dtype=manifest["dtype"]),
            cache=CacheConfig(
                block_size=block_size,
                num_blocks=manifest["usable_kv_tokens"] // block_size),
            scheduler=SchedulerConfig(
                max_num_seqs=256, max_num_batched_tokens=8192),
            execution=ExecutionConfig(mode="graph", device_index=0, max_gear=32),
            async_scheduling=manifest["async_scheduling"],
            async_batches=manifest["async_batches"])
        self._llm = LLM(config)
        self.runtime_kv_capacity = {
            "usable_tokens": manifest["usable_kv_tokens"],
            "runtime_block_size": block_size,
            "evidence": "constructed EngineConfig CacheConfig",
        }

    @property
    def prompt_ids(self) -> list[int]:
        return list(self._prompt_ids)

    def run(self, batch_size: int, output_tokens: int) -> list[list[int]]:
        return self._llm.generate(
            [list(self._prompt_ids) for _ in range(batch_size)],
            max_tokens=output_tokens)

    def close(self) -> None:
        self._llm.close()


class _VllmAdapter:
    def __init__(self, manifest: dict):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        self._manifest = manifest
        self._sampling_params = SamplingParams
        self._llm = LLM(
            model=manifest["model"], dtype=manifest["dtype"],
            trust_remote_code=True, max_model_len=manifest["max_model_len"],
            kv_cache_memory_bytes=manifest["kv_cache_memory_bytes"],
            enforce_eager=False, seed=manifest["seed"])
        self._prompt_ids = self._llm.get_tokenizer().encode(manifest["prompt"])
        engine = self._llm.llm_engine
        cache = engine.vllm_config.cache_config
        runtime_blocks = getattr(cache, "num_gpu_blocks", None)
        runtime_block_size = int(getattr(cache, "block_size"))
        if not isinstance(runtime_blocks, int) or runtime_blocks <= 0:
            raise ValueError("vLLM did not expose runtime num_gpu_blocks")
        self.runtime_kv_capacity = {
            "usable_tokens": runtime_blocks * runtime_block_size,
            "runtime_block_size": runtime_block_size,
            "runtime_blocks": runtime_blocks,
            "evidence": "vLLM cache_config runtime readback",
        }

    @property
    def prompt_ids(self) -> list[int]:
        return list(self._prompt_ids)

    def run(self, batch_size: int, output_tokens: int) -> list[list[int]]:
        params = self._sampling_params(
            max_tokens=output_tokens,
            temperature=self._manifest["temperature"],
            seed=self._manifest["seed"], ignore_eos=True)
        prompts = [
            {"prompt_token_ids": list(self._prompt_ids)}
            for _ in range(batch_size)]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    def close(self) -> None:
        shutdown = getattr(self._llm, "shutdown", None)
        if shutdown:
            shutdown()


def _adapter(manifest: dict, framework: str):
    if framework == "auto-infer":
        return _AutoInferAdapter(manifest)
    if framework == "omni-npu":
        prepare_omni_compatibility(manifest["model"])
    return _VllmAdapter(manifest)


def capture(manifest_path: Path, framework: str, output_dir: Path) -> None:
    import torch
    import torch_npu
    from torch.profiler import record_function
    from torch_npu.profiler import ProfilerActivity, profile

    manifest = load_manifest(manifest_path)
    workload = profile_configuration(manifest, framework)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{framework}.trace.json"
    metadata_path = output_dir / f"{framework}.metadata.json"
    engine = _adapter(manifest, framework)
    try:
        model_identity = _model_identity(manifest["model"], engine.prompt_ids)
        runtime_capacity = engine.runtime_kv_capacity
        if runtime_capacity["usable_tokens"] != workload["usable_kv_tokens"]:
            raise ValueError(
                "runtime KV capacity differs from profiling workload")
        for _ in range(workload["warmup_runs"]):
            engine.run(
                workload["batch_size"], workload["warmup_output_tokens"])
        torch.npu.synchronize()
        started_at = datetime.now(timezone.utc).isoformat()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            with record_function("qwen3/profiled_request"):
                with record_function(framework):
                    with record_function("qwen3/prefill_and_decode"):
                        outputs = engine.run(
                            workload["batch_size"], workload["output_tokens"])
            torch.npu.synchronize()
        completed_at = datetime.now(timezone.utc).isoformat()
        profiler.export_chrome_trace(str(trace_path))
    finally:
        engine.close()

    trace = validate_chrome_trace(trace_path)
    if len(outputs) != workload["batch_size"]:
        raise ValueError("profiled output batch length does not match workload")
    lengths = [len(tokens) for tokens in outputs]
    if any(length != workload["output_tokens"] for length in lengths):
        raise ValueError(f"profiled output lengths do not match workload: {lengths}")
    metadata = {
        "framework": framework,
        "trace": {
            "file": trace_path.name,
            "sha256": sha256_file(trace_path),
            **trace,
        },
        "workload": workload,
        "model_identity": model_identity,
        "runtime_kv_capacity": runtime_capacity,
        "environment": _environment(torch, torch_npu),
        "capture": {"started_at_utc": started_at,
                    "completed_at_utc": completed_at},
        "output_digest": token_digest(outputs[0]),
        "output_batch_digests": [token_digest(tokens) for tokens in outputs],
        "output_length": lengths[0],
    }
    write_profile_metadata(metadata_path, metadata)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("framework", choices=sorted(SUPPORTED_FRAMEWORKS))
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # omni-npu's patch loader reads argv[2] as a model compatibility slot.
    if args.framework == "omni-npu" and len(sys.argv) < 3:
        raise ValueError("omni-npu requires its plugin environment")
    capture(args.manifest, args.framework, args.output_dir)


if __name__ == "__main__":
    main()
