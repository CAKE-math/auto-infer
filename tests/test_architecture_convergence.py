"""Structural gates preventing retired compatibility paths from returning."""
import ast
import importlib.util
import inspect
from pathlib import Path

from auto_infer.engine.executor import RunnerExecutor
from auto_infer.layers.moe import fused_moe
from auto_infer.models.deepseek_v2 import DeepseekV2Model
from auto_infer.spec_decode import rejection_sampler
from auto_infer.worker import graph_decode_runner
from auto_infer.worker import graph_mtp_runner
from auto_infer.worker import decode_input_stager, staging
from auto_infer.worker.staging import splice_device_tokens


def _internal_imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_low_level_packages_do_not_import_orchestration_layers():
    package = Path(__file__).parents[1] / "auto_infer"
    violations = []
    forbidden = {
        "config": "auto_infer.engine",
        "layers": "auto_infer.worker",
        "distributed": "auto_infer.layers",
    }
    for subtree, prefix in forbidden.items():
        for path in (package / subtree).rglob("*.py"):
            for imported in _internal_imports(path):
                if imported == prefix or imported.startswith(prefix + "."):
                    violations.append(
                        f"{path.relative_to(package)} -> {imported}")
    assert violations == []


def test_model_package_boundary_stays_out_of_engine_and_runners():
    root = Path(__file__).parents[1]
    for relative in (
        "auto_infer/engine/engine_core.py",
        "auto_infer/worker/model_runner.py",
        "auto_infer/worker/graph_decode_runner.py",
        "auto_infer/worker/graph_mtp_runner.py",
    ):
        source = (root / relative).read_text()
        assert "model_package" not in source
        assert "auto_infer.harness" not in source


def test_production_modules_have_no_retired_implementations():
    assert not hasattr(DeepseekV2Model, "_forward_paged_legacy")
    assert not hasattr(graph_decode_runner, "_marshal_decode_batch")
    assert not hasattr(rejection_sampler, "emitted_tokens")
    assert "trace" not in inspect.signature(splice_device_tokens).parameters
    for name in (
        "_SpecGear",
        "_GraphMtpBackends",
    ):
        assert not hasattr(graph_mtp_runner, name)
    for name in ("_ensure_fused_fallback", "_fused_body", "_graph_decode"):
        assert not hasattr(graph_mtp_runner.GraphMtpPagedRunner, name)


def test_staging_primitives_have_one_owner():
    assert hasattr(staging, "splice_device_tokens")
    assert not hasattr(decode_input_stager, "splice_device_tokens")
    root = Path(__file__).parents[1]
    for relative in (
        "auto_infer/worker/decode_input_stager.py",
        "auto_infer/worker/prefill_input_stager.py",
        "auto_infer/worker/mtp_pipeline_stager.py",
    ):
        source = (root / relative).read_text()
        assert "upload_dirty_block_table(" in source
        assert "dirty_spans(" not in source


def test_mtp_runners_depend_on_attention_capability_not_gqa_concrete_types():
    root = Path(__file__).parents[1]
    for relative in (
        "auto_infer/worker/mtp_runner.py",
        "auto_infer/worker/graph_mtp_runner.py",
    ):
        source = (root / relative).read_text()
        assert "layers.attention.gqa" not in source
        assert "build_mtp_attention_backend" in source
    graph_source = (
        root / "auto_infer/worker/graph_mtp_runner.py").read_text()
    assert graph_source.count(
        "cos, sin = model._compute_cos_sin(gear.ppos)") == 1


def test_half_wired_modules_and_invalid_examples_are_absent():
    root = Path(__file__).parents[1]
    assert importlib.util.find_spec("auto_infer.models.deepseek_mtp") is None
    assert importlib.util.find_spec("auto_infer.pd.mooncake_transport") is None
    assert not (root / "auto_infer/profiling").exists()
    assert importlib.util.find_spec("auto_infer.engine.errors") is None
    assert not hasattr(fused_moe, "_fused_experts_ep_reference")
    for retired_script in (
        "mtp_pipeline_demo.py",
        "verify_fused_mtp_graph.py",
        "verify_fused_mtp_graph_batched.py",
    ):
        assert not (root / "scripts" / retired_script).exists()
    assert not (root / "auto_infer/deploy/v3_multinode.example.sh").exists()


def test_verification_helpers_do_not_live_in_production_packages():
    from auto_infer.distributed import parallel_state
    from auto_infer.worker import decode_epilogue

    assert importlib.util.find_spec("auto_infer.serving.router") is None
    assert importlib.util.find_spec("auto_infer.serving.sse_client") is None
    assert not hasattr(fused_moe, "build_expert_weights_w8a8")
    assert not hasattr(fused_moe, "fused_experts_w8a8")
    assert not hasattr(parallel_state, "ep_all_reduce")
    assert not hasattr(decode_epilogue, "DecodeEpilogue")
    assert hasattr(decode_epilogue, "is_capturable_greedy")


def test_runner_executor_delegates_the_execution_protocol():
    class Runner:
        def supports_async(self):
            return True

        def prepare(self, plan, previous):
            return "prepared", plan, previous

        def submit_prepared(self, prepared):
            return "submitted", prepared

        def submit(self, plan, previous):
            return plan, previous

        def sampled_of(self, handle):
            return "sampled", handle

        def collect(self, handle):
            return "collected", handle

        def collect_async(self, handle):
            return "future", handle

        def collect_result(self, future):
            return "result", future

        def execute(self, plan):
            return "executed", plan

        def execute_spec_mtp(self, plan):
            return "mtp", plan

        def close(self):
            self.closed = True

    runner = Runner()
    executor = RunnerExecutor(runner)
    assert executor.supports_async()
    assert executor.prepare("plan", "previous") == (
        "prepared", "plan", "previous")
    assert executor.submit_prepared("prepared") == (
        "submitted", "prepared")
    assert executor.submit("plan", "previous") == ("plan", "previous")
    assert executor.sampled_of("handle") == ("sampled", "handle")
    assert executor.collect("handle") == ("collected", "handle")
    assert executor.collect_async("handle") == ("future", "handle")
    assert executor.collect_result("future") == ("result", "future")
    assert executor.execute("plan") == ("executed", "plan")
    assert executor.execute_spec_mtp("plan") == ("mtp", "plan")
    executor.close()
    assert runner.closed
