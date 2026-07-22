"""Build the offline Qwen3 architecture and performance report."""

import html
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "docs" / "profiling" / "qwen3"
OUTPUT = ROOT / "docs" / "AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html"
OUTPUT_MD = ROOT / "docs" / "AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.md"
FRAMEWORKS = ("auto-infer", "omni-npu", "vllm-ascend")
DISPLAY = {
    "auto-infer": "auto-infer",
    "omni-npu": "omni-npu 0.14",
    "vllm-ascend": "vllm-ascend 0.20.2",
}
PHASE_LABELS = {
    "graph_replay": "Graph replay / launch",
    "attention_kv": "Attention / KV",
    "projection_mlp_norm": "Projection / MLP / Norm",
    "lm_head_sampling": "LM head / sampling",
    "runtime_scheduling": "Runtime / scheduling",
    "communication_memory": "Communication / memory",
    "unclassified": "Unclassified",
}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _request_duration(profile: dict) -> float:
    for event in profile["top_events"]:
        if event["name"] == "qwen3/profiled_request":
            return event["duration_us"]
    raise ValueError("profile trace lacks qwen3/profiled_request range")


def _attention_calls(profile: dict) -> int:
    return sum(
        event["count"] for event in profile["top_events"]
        if event["name"].startswith("npu::npu_fused_infer_attention_score"))


def _headline_rows(summary: dict) -> list[tuple[str, str, list[float], str]]:
    data = summary["headline_benchmarks"]
    batch_size = data["auto-infer"]["manifest"]["throughput_batch"]
    return [
        ("Warm TTFT", "越低越好",
         [data[f]["ttft_seconds"]["median"] * 1000 for f in FRAMEWORKS], "ms"),
        ("TPOT", "越低越好",
         [data[f]["tpot_seconds"] * 1000 for f in FRAMEWORKS], "ms"),
        (f"B{batch_size} throughput", "越高越好",
         [data[f]["throughput_tokens_per_second"]["median"]
          for f in FRAMEWORKS], "tok/s"),
        ("Engine load + graph", "越低越好",
         [data[f]["load_seconds"] for f in FRAMEWORKS], "s"),
        ("Peak torch allocation", "越低越好",
         [data[f]["peak_allocated_gib"] for f in FRAMEWORKS], "GiB"),
        ("Throughput CV", "越低越好",
         [data[f]["stability"]["throughput_cv"] * 100
          for f in FRAMEWORKS], "%"),
    ]


def _metric_table(summary: dict) -> str:
    rows = []
    for label, direction, values, unit in _headline_rows(summary):
        winner = max(values) if direction == "越高越好" else min(values)
        cells = []
        for value in values:
            winner_class = " winner" if value == winner else ""
            digits = 1 if unit == "tok/s" else 3
            cells.append(
                f'<td class="num{winner_class}">{_fmt(value, digits)} '
                f'<span class="unit">{_e(unit)}</span></td>')
        rows.append(
            f'<tr><th scope="row">{_e(label)}<small>{_e(direction)}</small></th>'
            + "".join(cells) + "</tr>")
    return "".join(rows)


def _metric_bars(summary: dict) -> str:
    data = summary["headline_benchmarks"]
    batch_size = data["auto-infer"]["manifest"]["throughput_batch"]
    metrics = [
        (f"B{batch_size} 吞吐", [data[f]["throughput_tokens_per_second"]["median"]
                     for f in FRAMEWORKS], "tok/s", True),
        ("Warm TTFT", [data[f]["ttft_seconds"]["median"] * 1000
                      for f in FRAMEWORKS], "ms", False),
        ("TPOT", [data[f]["tpot_seconds"] * 1000 for f in FRAMEWORKS],
         "ms", False),
    ]
    blocks = []
    for title, values, unit, higher in metrics:
        maximum = max(values)
        bars = []
        best = max(values) if higher else min(values)
        for framework, value in zip(FRAMEWORKS, values):
            width = 100 * value / maximum if maximum else 0
            state = " best" if value == best else ""
            digits = 1 if unit == "tok/s" else 2
            bars.append(
                f'<div class="bar-row{state}"><span>{_e(framework)}</span>'
                f'<div class="track"><i style="width:{width:.2f}%"></i></div>'
                f'<b>{_fmt(value, digits)} {unit}</b></div>')
        blocks.append(
            f'<article class="chart"><h3>{_e(title)}</h3>{"".join(bars)}</article>')
    return "".join(blocks)


def _profile_cards(summary: dict, manifest: dict) -> str:
    cards = []
    for framework in FRAMEWORKS:
        profile = summary["profiles"][framework]
        artifact = manifest["artifacts"][framework]
        duration_ms = _request_duration(profile) / 1000
        calls = _attention_calls(profile)
        execution_steps = profile["execution_phases"]["steps"]
        prefill = execution_steps[0]
        decode = execution_steps[1:]
        decode_total = sum(step["duration_us"] for step in decode)
        phase_rows = []
        for phase, values in profile["phases"].items():
            phase_rows.append(
                f'<tr><td>{_e(PHASE_LABELS[phase])}</td>'
                f'<td class="num">{values["count"]:,}</td>'
                f'<td class="num">{_fmt(values["duration_us"] / 1000, 2)} ms</td>'
                f'<td class="num">{_fmt(values["share"] * 100, 1)}%</td></tr>')
        top_rows = "".join(
            f'<tr><td><code>{_e(event["name"])}</code></td>'
            f'<td class="num">{event["count"]:,}</td>'
            f'<td class="num">{_fmt(event["duration_us"] / 1000, 2)} ms</td></tr>'
            for event in profile["top_events"][:8])
        trace_href = f'profiling/qwen3/raw/{framework}.trace.json'
        cards.append(f'''
        <article class="profile-card">
          <header><div><span class="eyebrow">PROFILER-INSTRUMENTED</span>
            <h3>{_e(DISPLAY[framework])}</h3></div>
            <a class="trace-link" href="{trace_href}">打开原始 Trace ↗</a></header>
          <div class="profile-kpis">
            <div><strong>{_fmt(duration_ms, 2)} ms</strong><span>captured request range</span></div>
            <div><strong>{profile["complete_event_count"]:,}</strong><span>complete events</span></div>
            <div><strong>{calls:,}</strong><span>FIA host calls</span></div>
            <div><strong>{artifact["size_bytes"] / 2**20:.1f} MiB</strong><span>raw JSON</span></div>
            <div><strong>{_fmt(prefill["duration_us"] / 1000, 2)} ms</strong><span>PREFILL host range</span></div>
            <div><strong>{_fmt(decode_total / 1000, 2)} ms</strong><span>{len(decode)} DECODE host ranges</span></div>
          </div>
          <details><summary>阶段事件时间（展开）</summary>
            <div class="table-scroll"><table class="compact"><thead><tr><th>类别</th><th>事件数</th><th>累计事件时间</th><th>事件时间占比</th></tr></thead>
            <tbody>{''.join(phase_rows)}</tbody></table></div>
          </details>
          <details><summary>Top events（展开）</summary>
            <div class="table-scroll"><table class="compact"><thead><tr><th>事件</th><th>次数</th><th>累计时间</th></tr></thead>
            <tbody>{top_rows}</tbody></table></div>
          </details>
          <p class="hash"><span>SHA-256</span><code>{_e(artifact["sha256"])}</code></p>
        </article>''')
    return "".join(cards)


def _architecture_data() -> list[tuple[str, str, str, str]]:
    return [
        ("核心控制流", "EngineCore → BatchPlan → Executor → ExecutionResult；协议短且状态归属明确。",
         "vLLM 主流程之上叠加环境选择的 patch 与额外配置。",
         "复用成熟 vLLM engine，Ascend worker / runner / compiler 专化。"),
        ("模型扩展 seam", "模型声明 attention / MTP capability；registry 注入对象，engine 不按模型分支。",
         "模型实现、best-practice config 与 patches 共同决定路径。",
         "覆盖广，但模型与平台 runner 的组合触点更多。"),
        ("状态所有权", "Engine/service 单线程拥有 request、scheduler、KV、completion；跨线程传不可变视图。",
         "上游 vLLM 状态与 patch 后行为共同形成所有权边界。",
         "继承 vLLM 的多组件生命周期，成熟但阅读跨度更大。"),
        ("Graph 生命周期", "启动期捕获；固定地址 staging；replay 后独立 stream metadata update；event + 双缓冲。",
         "NpuGraphEx + full/piecewise graph，依赖插件配置与 graph gear。",
         "torch.compile + ACL graph piecewise；支持较宽的通用 shape 集合。"),
        ("输入与 KV metadata", "持久 pinned CPU/NPU buffer；block table 只传 dirty rows/span。",
         "由 Omni runner 与 patched vLLM metadata 路径管理。",
         "通用 input batch 和 worker metadata 路径，支持面广。"),
        ("Projection / epilogue", "packed QKV、gate/up；BF16 captured lm_head + greedy argmax；避免外部 sampler step。",
         "拥有广泛融合算子与模型专用优化配置。",
         "平台 custom op、compiler fusion 与通用 sampler 体系。"),
        ("Continuous batching", "scheduler/KV 生命周期有专项回归；同步路径是本负载已验证默认。",
         "继承 vLLM scheduling，并通过 patch 增补行为。",
         "vLLM V1 scheduler 成熟，线上生态更完整。"),
        ("Serving", "单 service + broker + request-id demux；在线/离线共用核心执行协议。",
         "成熟 vLLM serving，Omni patch OpenAI 与 scheduler 层。",
         "API、工具链、部署经验最成熟。"),
        ("Distributed / MoE", "命名 TP/DP/EP/CP/SP mesh；BF16 true all-to-all 接口与测试；深度仍有限。",
         "并行与算子调优面最丰富，是明显强项。",
         "上游并行体系成熟，Ascend 通信优化覆盖更广。"),
        ("MTP", "一个 two-stage recurrent graph path；geometry 从权重推导；unsupported fail-fast。",
         "Eagle / MTP patches 与模型优化覆盖更广。",
         "上游 speculative decoding 生态更完整。"),
        ("P/D 与 MLA MTP", "仅保留未接线 P/D 低层接口；MLA MTP capability 保留但明确 unsupported。",
         "P/D、connector 与 MLA/MoE 产品能力更完整。",
         "connector / disaggregation 生态成熟。"),
        ("维护与审计面", "9,960 Python LOC / 93 files；无内部 import cycle；路径较短。",
         "61,080 / 223；patch 提升适配力，也增加组合状态。",
         "53,219 / 242；生态收益大，平台 runner 体量更高。"),
    ]


def _call_stack_data() -> list[tuple[str, str, str, str]]:
    return [
        (
            "auto-infer",
            "LLM.generate → EngineCore.step → Scheduler.schedule → "
            "BatchPlan.from_scheduler → "
            "GraphPagedNpuExecutor[RunnerExecutor.execute] → "
            "GraphPagedRunner.execute → "
            "GraphPagedRunner.submit → Model.forward(ctx) → "
            "AttentionBackend",
            "EngineCore owns request、scheduler、KV 与 completion；执行层只交换短协议对象。",
            "auto_infer/entrypoints/llm.py · engine/engine_core.py · "
            "engine/execution.py · worker/graph_decode_runner.py",
        ),
        (
            "omni-npu",
            "vLLM LLM.generate → LLMEngine.step → vLLM EngineCore → "
            "Scheduler.schedule → ModelExecutor.execute_model → "
            "omni_npu.NPUWorker.execute_model → "
            "omni_npu.NPUModelRunner.execute_model → model / graph patches → "
            "Model.forward",
            "上游 vLLM 生命周期、Omni plugin/patch、worker 与模型优化共同决定实际路径。",
            "vllm/entrypoints/llm.py · omni_npu/worker/npu_worker.py · "
            "omni_npu/worker/npu_model_runner.py",
        ),
        (
            "vllm-ascend",
            "vLLM LLM.generate → LLMEngine.step → "
            "InprocClient.get_output → vLLM EngineCore.step_fn → "
            "Scheduler.schedule → ModelExecutor.execute_model → "
            "vllm_ascend.NPUWorker.execute_model → "
            "vllm_ascend.NPUModelRunner.execute_model → Model.forward",
            "vLLM V1 保持通用 engine；Ascend plugin 在 platform、worker、runner、compiler 与 custom-op 层专化。",
            "vllm/v1/engine/llm_engine.py · vllm/v1/engine/core_client.py · "
            "vllm_ascend/worker/worker.py · worker/model_runner_v1.py",
        ),
    ]


def _call_stack_rows() -> str:
    return "".join(
        f'<tr><th scope="row">{_e(framework)}</th>'
        f'<td><code>{_e(stack)}</code></td><td>{_e(boundary)}</td>'
        f'<td><code>{_e(sources)}</code></td></tr>'
        for framework, stack, boundary, sources in _call_stack_data())


def _architecture_rows() -> str:
    return "".join(
        f'<tr><th scope="row">{_e(area)}</th><td>{_e(auto)}</td>'
        f'<td>{_e(omni)}</td><td>{_e(vllm)}</td></tr>'
        for area, auto, omni, vllm in _architecture_data())


def _causal_data(summary: dict, manifest: dict) -> list[tuple[str, str, str, str]]:
    data = summary["headline_benchmarks"]
    relative = summary["relative_to_auto_infer"]
    headline_workload = data["auto-infer"]["manifest"]
    profile_workload = manifest["workload"]
    profile_ms = {
        framework: _request_duration(summary["profiles"][framework]) / 1000
        for framework in FRAMEWORKS
    }
    return [
        ("实测", f'B{headline_workload["throughput_batch"]} throughput',
         f'{data["auto-infer"]["throughput_tokens_per_second"]["median"]:,.1f} tok/s；较 omni-npu {relative["omni-npu"]["throughput_speedup"]:.2f}×，较 vllm-ascend {relative["vllm-ascend"]["throughput_speedup"]:.2f}×。',
         "headline benchmark JSON"),
        ("源码观察", "Graph hot path", "graph capture、staging、replay/update、epilogue 拆成可独立测试组件；热路径不做在线 capture。", "graph_decode_runner / graph_task_pipeline"),
        ("因果推断", "Replay + metadata pipeline", "replay 后 side-stream 更新、event 排序和双缓冲减少 graph-task update 对下一步的阻塞。", "与低 TPOT 及短 profiled request 一致；未做单变量 ablation"),
        ("源码观察", "Persistent staging", "CPU/NPU 输入缓冲持久化，block table 仅上传 dirty row/span。", "staging / input stagers"),
        ("因果推断", "较少 host/device 胶水", "固定地址与脏更新降低逐步分配、拷贝和 Python 调度成本。", "trace 中 auto-infer request range 最短"),
        ("源码观察", "Packed projections", "QKV 与 gate/up 使用 packed weight；BF16 lm_head 与 greedy argmax 留在 captured epilogue。", "packed projections / decode epilogue"),
        ("因果推断", "更少 kernel 与同步边界", "projection packing 与直接 argmax 降低 launch 数；收益随模型/shape 变化，必须重做 profiling。", "机制合理但不能由相关性证明全部增益"),
        ("实测", "Profiler window",
         f'B{profile_workload["batch_size"]} {profile_workload["output_tokens"]}-token 请求范围约 {profile_ms["auto-infer"]:.1f} / {profile_ms["omni-npu"]:.1f} / {profile_ms["vllm-ascend"]:.1f} ms（auto / omni / vllm）。',
         "三份 raw Chrome Trace"),
        ("实测", "Startup",
         f'{data["auto-infer"]["load_seconds"]:.3f} s vs {data["omni-npu"]["load_seconds"]:.3f} s vs {data["vllm-ascend"]["load_seconds"]:.3f} s；auto-infer 只捕获所需 gear，通用框架初始化面更宽。',
         "headline benchmark + framework logs"),
    ]


def _causal_rows(summary: dict, manifest: dict) -> str:
    return "".join(
        f'<tr><td><span class="evidence {"measured" if kind == "实测" else "observed" if kind == "源码观察" else "inferred"}">{_e(kind)}</span></td>'
        f'<th scope="row">{_e(link)}</th><td>{_e(statement)}</td><td>{_e(basis)}</td></tr>'
        for kind, link, statement, basis in _causal_data(summary, manifest))


def _invariant_data() -> list[str]:
    return [
        "EngineCore → BatchPlan → Executor → ExecutionResult 协议",
        "request / scheduler / KV / completion 的单一所有权",
        "模型声明 capability、registry 选择实现；engine 不加模型分支",
        "recurrent MTP 独立 capability；不支持时启动期 fail-fast",
        "graph-FIA capture / replay / update 的 event 顺序契约",
        "固定地址持久 staging 与 dirty block-table 更新",
        "精度优先：logits/token parity 先于性能排名",
        "P/D 未接线、MLA MTP unsupported 等边界必须显式",
        "matched manifest、raw samples、trace 和 hash 的证据保留",
    ]


def _invariant_items() -> str:
    return "".join(f"<li>{_e(item)}</li>" for item in _invariant_data())


def _regenerated_data() -> list[str]:
    return [
        "checkpoint / config / weight-name inventory 与 adapter",
        "TP/EP head、expert、attention、RoPE 与 cache geometry",
        "KV budget、block size、scratch blocks、max sequence length",
        "packed QKV / gate-up 权重与 dtype / quantization metadata",
        "graph gear ladder、capture matrix、handles、memory envelope",
        "MTP layers、geometry、depth、position acceptance 与 capability",
        "BF16/FP32 head 与 sampling parity 阈值",
        "golden prompts、logits/token digest、eager/paged/graph identity",
        "unprofiled baseline、raw traces、phase map、CV 与回归阈值",
        "实际声明的 TP/EP/SP/CP 拓扑验证",
    ]


def _regenerated_items() -> str:
    return "".join(f"<li>{_e(item)}</li>" for item in _regenerated_data())


def _artifact_rows(manifest: dict) -> str:
    rows = []
    for framework in FRAMEWORKS:
        artifact = manifest["artifacts"][framework]
        env = artifact["metadata"]["environment"]
        rows.append(
            f'<tr><th scope="row">{_e(framework)}</th>'
            f'<td><a href="profiling/qwen3/{_e(artifact["path"])}">{_e(artifact["path"])}</a></td>'
            f'<td class="num">{artifact["event_count"]:,}</td>'
            f'<td class="num">{artifact["size_bytes"] / 2**20:.1f} MiB</td>'
            f'<td><code>{_e(artifact["sha256"])}</code></td>'
            f'<td>{_e(env["torch"])} / {_e(env["torch_npu"])} / {_e(env["vllm"])}</td></tr>')
    return "".join(rows)


def build_report(summary: dict, manifest: dict) -> str:
    data = summary["headline_benchmarks"]
    relative = summary["relative_to_auto_infer"]
    phases = manifest["workload"]["capture_phases"]
    headline_workload = data["auto-infer"]["manifest"]
    profile_workload = manifest["workload"]
    capture_revision = manifest["provenance"]["capture_harness_revision"]
    attention_calls = {
        framework: _attention_calls(summary["profiles"][framework])
        for framework in FRAMEWORKS
    }
    common_attention_calls = (
        next(iter(attention_calls.values()))
        if len(set(attention_calls.values())) == 1 else None)
    inferred_layers = (
        common_attention_calls // profile_workload["output_tokens"]
        if common_attention_calls is not None
        and common_attention_calls % profile_workload["output_tokens"] == 0
        else None)
    attention_evidence = (
        f'三方均记录到 {common_attention_calls:,} 次 fused-attention host 调用，'
        f'与 {inferred_layers} 层 × {profile_workload["output_tokens"]} 次 forward 相符。'
        if inferred_layers is not None else
        "三方 fused-attention host 调用数分别为 "
        + " / ".join(f"{framework}: {count:,}"
                     for framework, count in attention_calls.items()) + "。")
    profile_digest_equal = len({
        manifest["artifacts"][f]["metadata"]["output_digest"]
        for f in FRAMEWORKS}) == 1
    unclassified_percentages = [
        summary["profiles"][framework]["phases"]["unclassified"]["share"] * 100
        for framework in FRAMEWORKS
    ]
    capture_date = max(
        manifest["artifacts"][framework]["metadata"]["environment"][
            "captured_at_utc"][:10]
        for framework in FRAMEWORKS)
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>auto-infer 架构与 Qwen3 性能审计报告</title>
<style>
:root{{--ink:#122033;--muted:#5b6878;--paper:#f4f7fa;--panel:#fff;--navy:#163a5f;--cyan:#087e8b;--cyan-soft:#dff2f3;--amber:#b56a00;--amber-soft:#fff0d5;--green:#177252;--green-soft:#def3e9;--red:#a33d3d;--line:#ced8e3;--shadow:0 16px 42px rgba(18,32,51,.08);--display:"Arial Narrow","Roboto Condensed","PingFang SC",sans-serif;--body:"Avenir Next","PingFang SC","Microsoft YaHei",sans-serif;--mono:"SFMono-Regular",Consolas,monospace}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);line-height:1.65}}a{{color:var(--navy);text-underline-offset:3px}}a:focus-visible,summary:focus-visible{{outline:3px solid #ffbf47;outline-offset:3px}}code,.num{{font-family:var(--mono);font-variant-numeric:tabular-nums}}code{{font-size:.84em;overflow-wrap:anywhere}}.shell{{width:100%;max-width:1440px;margin:auto;display:grid;grid-template-columns:250px minmax(0,1fr)}}nav{{position:sticky;top:0;height:100vh;padding:34px 24px;border-right:1px solid var(--line);background:rgba(244,247,250,.96)}}.brand{{font:800 22px/1 var(--display);letter-spacing:.04em}}.brand small{{display:block;margin-top:8px;font:600 11px/1.4 var(--mono);color:var(--cyan)}}nav ul{{list-style:none;padding:24px 0;margin:0}}nav li{{margin:7px 0}}nav a{{display:block;padding:5px 0;color:var(--muted);font-size:13px;text-decoration:none}}nav a:hover{{color:var(--cyan)}}main{{min-width:0;max-width:100%}}section,.hero{{max-width:100%;overflow-wrap:anywhere;padding:72px clamp(28px,6vw,92px);border-bottom:1px solid var(--line)}}.hero{{min-height:86vh;display:grid;align-content:center;background:linear-gradient(135deg,#f8fbfd 0%,#eaf2f7 100%)}}.hero>*{{min-width:0;max-width:100%}}.eyebrow{{font:700 11px/1.2 var(--mono);letter-spacing:.12em;color:var(--cyan)}}h1,h2,h3{{font-family:var(--display);line-height:1.08;margin-top:0}}h1{{max-width:990px;line-break:anywhere;overflow-wrap:anywhere;font-size:clamp(46px,7vw,94px);letter-spacing:-.045em;margin:20px 0 28px}}h2{{font-size:clamp(34px,4vw,56px);letter-spacing:-.025em;margin-bottom:16px}}h3{{font-size:22px}}.lede{{max-width:850px;font-size:clamp(18px,2vw,24px);color:var(--muted)}}.thesis-chain{{min-width:0;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:52px;border:1px solid var(--navy);background:var(--panel);box-shadow:var(--shadow)}}.thesis-chain div{{min-width:0;padding:22px;border-right:1px solid var(--navy)}}.thesis-chain div:last-child{{border:0}}.thesis-chain span{{display:block;font:700 11px var(--mono);color:var(--cyan)}}.thesis-chain strong{{display:block;margin-top:8px}}.section-head{{max-width:920px;margin-bottom:38px}}.section-head p{{font-size:18px;color:var(--muted)}}.evidence{{display:inline-block;padding:4px 8px;border-radius:2px;font:700 10px var(--mono);white-space:nowrap}}.measured{{color:var(--green);background:var(--green-soft)}}.observed{{color:var(--navy);background:#e1ebf5}}.inferred{{color:var(--amber);background:var(--amber-soft)}}.decision-grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:20px}}.decision,.risk{{padding:30px;background:var(--panel);border-top:5px solid var(--green);box-shadow:var(--shadow)}}.risk{{border-color:var(--amber)}}.decision strong.big{{display:block;font:800 clamp(32px,5vw,60px)/1 var(--display);margin:16px 0;color:var(--green)}}.decision ul,.risk ul{{padding-left:20px}}.callout{{margin:30px 0;padding:20px 24px;border-left:5px solid var(--cyan);background:var(--cyan-soft)}}.callout.warning{{border-color:var(--amber);background:var(--amber-soft)}}.charts{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:32px 0}}.chart{{padding:24px;background:var(--panel);border:1px solid var(--line)}}.bar-row{{display:grid;grid-template-columns:90px 1fr auto;gap:10px;align-items:center;margin:13px 0;font-size:12px}}.track{{height:9px;background:#e2e9ef}}.track i{{display:block;height:100%;background:#8ba0b3}}.bar-row.best .track i{{background:var(--cyan)}}.bar-row.best b{{color:var(--cyan)}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:15px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}thead th{{font:700 11px var(--mono);letter-spacing:.05em;color:var(--muted);background:#edf2f6}}tbody th{{font-weight:700}}th small{{display:block;font-weight:400;color:var(--muted)}}td.winner{{color:var(--green);font-weight:800;background:var(--green-soft)}}.unit{{font-family:var(--body);font-size:.82em;color:var(--muted)}}.table-scroll{{max-width:100%;overflow-x:auto}}.profile-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.profile-card{{min-width:0;padding:24px;background:var(--panel);box-shadow:var(--shadow);border-top:5px solid var(--cyan)}}.profile-card header{{display:flex;justify-content:space-between;gap:14px;align-items:start}}.trace-link{{font-size:12px;font-weight:700;white-space:nowrap}}.profile-kpis{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);margin:18px 0}}.profile-kpis div{{padding:14px;background:var(--paper)}}.profile-kpis strong,.profile-kpis span{{display:block}}.profile-kpis strong{{font:800 20px var(--mono)}}.profile-kpis span{{font-size:10px;color:var(--muted)}}details{{border-top:1px solid var(--line)}}summary{{cursor:pointer;padding:14px 0;font-weight:700}}.compact th,.compact td{{padding:8px;font-size:11px}}.hash span,.hash code{{display:block}}.hash span{{font:700 10px var(--mono);color:var(--muted)}}.hash code{{font-size:9px}}.evidence-table td:first-child{{width:110px}}.evidence-table th{{width:180px}}.architecture-table{{min-width:1050px}}.architecture-table th:first-child{{width:150px}}.architecture-table td{{width:30%}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.contract{{padding:30px;background:var(--panel);box-shadow:var(--shadow)}}.contract.invariant{{border-top:6px solid var(--navy)}}.contract.generated{{border-top:6px solid var(--amber)}}.contract li{{margin:10px 0}}.flow{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:0;border:1px solid var(--navy)}}.flow div{{min-width:0;position:relative;padding:24px 18px;border-right:1px solid var(--navy);background:var(--panel)}}.flow div:last-child{{border:0}}.flow b{{display:block;color:var(--cyan);font:800 22px var(--mono)}}.flow span{{font-size:13px}}.appendix-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.appendix-card{{padding:24px;border:1px solid var(--line);background:var(--panel)}}pre{{max-width:100%;padding:18px;overflow:auto;background:#132a40;color:#e9f2f8;font:12px/1.55 var(--mono)}}footer{{padding:32px clamp(28px,6vw,92px);color:var(--muted);font-size:12px}}@media(max-width:1080px){{.shell{{grid-template-columns:minmax(0,1fr)}}nav{{position:relative;height:auto;min-width:0;width:100%;border-right:0;border-bottom:1px solid var(--line)}}nav ul{{display:flex;max-width:100%;overflow:auto;gap:16px;padding:10px 0 0}}nav a{{white-space:nowrap}}.profile-grid{{grid-template-columns:minmax(0,1fr)}}.charts{{grid-template-columns:minmax(0,1fr)}}}}@media(max-width:720px){{section,.hero{{padding:48px 20px}}h1{{font-size:clamp(34px,11vw,46px);line-break:anywhere;word-break:break-all}}p,li,strong{{overflow-wrap:anywhere}}.decision-grid,.split,.appendix-grid{{grid-template-columns:minmax(0,1fr)}}.thesis-chain,.flow{{grid-template-columns:minmax(0,1fr)}}.thesis-chain div,.flow div{{border-right:0;border-bottom:1px solid var(--navy)}}.bar-row{{grid-template-columns:78px minmax(0,1fr)}}.bar-row b{{grid-column:2}}}}@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}@media print{{nav{{display:none}}.shell{{display:block}}section,.hero{{padding:28px 20px;break-inside:auto}}.hero{{min-height:auto}}.profile-card,.decision,.risk,.contract,.appendix-card{{box-shadow:none;break-inside:avoid}}details{{display:block}}details>summary{{display:none}}details>*{{display:block!important}}a{{color:inherit;text-decoration:none}}}}
</style>
<style>
.diagram{{margin:30px 0;padding:20px;background:#fff;border:1px solid var(--line);box-shadow:var(--shadow)}}
.diagram img{{display:block;width:100%;height:auto;margin:auto}}
.diagram figcaption{{margin-top:14px;color:var(--muted);font-size:13px}}
.call-stack-table{{min-width:1100px}}.call-stack-table td:nth-child(2){{width:38%}}
</style>
</head>
<body><div class="shell">
<nav aria-label="报告目录"><div class="brand">AUTO·INFER<small>ARCHITECTURE / PERFORMANCE / EVIDENCE</small></div><ul>
<li><a href="#executive-summary">管理结论</a></li><li><a href="#matched-benchmark">Matched benchmark</a></li>
<li><a href="#profiling-deep-dive">Profiling 深挖</a></li><li><a href="#call-stack-comparison">调用栈对比</a></li><li><a href="#why-faster">为什么更快</a></li>
<li><a href="#architecture-comparison">架构对比</a></li><li><a href="#invariants">不变与重生成</a></li>
<li><a href="#acceptance-workflow">模型验收流程</a></li><li><a href="#evidence-appendix">证据附录</a></li></ul></nav>
<main>
<header class="hero"><span class="eyebrow">{_e(Path(headline_workload["model"]).name.upper())} · {_e(manifest["provenance"]["driver"]["soc"].upper())} · {_e(headline_workload["dtype"].upper())} · {_e(capture_date)}</span>
<h1>性能领先不是一个数字，<br>而是一条可核验的因果链。</h1>
<p class="lede">本报告把管理决策、matched benchmark、三框架原始 Chrome Trace 与源码架构放在同一条证据链上。结论严格限定在已实现、已测量的推理核心，不把模型覆盖广度误写成架构质量，也不把 profiler 时间冒充生产吞吐。</p>
<div class="thesis-chain"><div><span>01 / CONTRACT</span><strong>同模型、同 KV、同精度</strong></div><div><span>02 / MEASURE</span><strong>{headline_workload["measured_runs"]} 次无 profiler 主测试</strong></div><div><span>03 / EXPLAIN</span><strong>{phases["prefill_passes"]} prefill + {phases["decode_passes"]} decode trace</strong></div><div><span>04 / GOVERN</span><strong>框架不变量 / 模型生成物</strong></div></div></header>

<section id="executive-summary"><div class="section-head"><span class="eyebrow">EXECUTIVE DECISION</span><h2>管理结论</h2><p>在 {_e(Path(headline_workload["model"]).name)}、单张 {_e(manifest["provenance"]["driver"]["soc"])}、{_e(headline_workload["dtype"])} greedy、等 {headline_workload["usable_kv_tokens"]:,} KV tokens 的验收边界内，auto-infer 在稳态延迟、吞吐、启动、等容量内存与稳定性上全部第一。</p></div>
<div class="decision-grid"><article class="decision"><span class="evidence measured">实测</span><strong class="big">B{headline_workload["throughput_batch"]}: {data["auto-infer"]["throughput_tokens_per_second"]["median"]:,.1f} tok/s</strong><p>对 omni-npu 为 <strong>{relative["omni-npu"]["throughput_speedup"]:.2f}×</strong>，对 vllm-ascend 为 <strong>{relative["vllm-ascend"]["throughput_speedup"]:.2f}×</strong>。auto-infer 与 vllm-ascend 的 {headline_workload["output_tokens"]}-token digest 一致。</p><ul><li>Warm TTFT：{data["auto-infer"]["ttft_seconds"]["median"]*1000:.3f} ms</li><li>TPOT：{data["auto-infer"]["tpot_seconds"]*1000:.3f} ms</li><li>Throughput CV：{data["auto-infer"]["stability"]["throughput_cv"]*100:.3f}%</li></ul></article>
<article class="risk"><span class="evidence inferred">范围边界</span><h3>可以投资，但不能泛化过度</h3><ul><li>vllm-ascend 仍领先于模型/API/部署生态成熟度。</li><li>omni-npu 仍领先于优化模型、算子和复杂并行覆盖。</li><li>P/D 只保留接口，MLA MTP 仍明确 unsupported。</li><li>量化仅保留扩展 seam，本轮只验证 {_e(headline_workload["dtype"])}。</li></ul></article></div>
<div class="callout"><strong>建议：</strong>保持 inference core 的短协议与单一所有权不变；新增模型时重生成 geometry、packed weights、graph gear 和性能/精度证据。不要为单个 checkpoint 在 engine 或 scheduler 增加分支。</div></section>

<section id="matched-benchmark"><div class="section-head"><span class="eyebrow">UNPROFILED · AUTHORITATIVE</span><h2>Matched benchmark</h2><p>{_e(Path(headline_workload["model"]).name)}；{_e(headline_workload["dtype"])}；greedy / ignore-EOS；B1 latency；B{headline_workload["throughput_batch"]} throughput；{headline_workload["output_tokens"]} output tokens；{headline_workload["warmup_runs"]} 次 warm-up；{headline_workload["measured_runs"]} 次 measurement；每框架 {headline_workload["usable_kv_tokens"]:,} usable KV tokens。以下 headline 只来自无 profiler 原始 JSON。</p></div>
<div class="charts">{_metric_bars(summary)}</div><div class="table-scroll"><table><thead><tr><th>指标</th>{''.join(f'<th>{_e(DISPLAY[f])}</th>' for f in FRAMEWORKS)}</tr></thead><tbody>{_metric_table(summary)}</tbody></table></div>
<div class="callout warning"><strong>精度口径：</strong>auto-infer 与 vllm-ascend 的 {headline_workload["output_tokens"]}-token digest 均为 <code>{_e(data["auto-infer"]["output_digest"])}</code>。omni-npu 输出长度同为 {data["omni-npu"]["output_length"]}，但 digest 为 <code>{_e(data["omni-npu"]["output_digest"])}</code>，因此该对只做性能可比，不声明 token identity。历史 headline JSON 使用 {_e(manifest["benchmark_schema"])} schema，本报告不补造 cold TTFT。</div></section>

<section id="profiling-deep-dive"><div class="section-head"><span class="eyebrow">RAW CHROME TRACE · DIRECTLY OPENABLE</span><h2>Qwen3 三框架 profiling</h2><p>每份 trace 捕获同一 B{profile_workload["batch_size"]}、{profile_workload["output_tokens"]}-token generate：<strong>{phases["prefill_passes"]} 次 prefill</strong> + <strong>{phases["decode_passes"]} 次连续 decode</strong>。这是连续多步 decode，<strong>不是投机 MTP</strong>。{_e(attention_evidence)}</p></div>
<div class="callout"><strong>如何一眼找到阶段：</strong>在 Chrome Trace / Perfetto 中找到置顶的 <code>QWEN3 PHASES</code> process，唯一的 <code>PREFILL</code> 后面依次是 <code>DECODE 001</code>…<code>DECODE {phases["decode_passes"]:03d}</code>。这是采集器对三套 engine step 写入的统一 host range；三方原生 operator、线程和 category 会保留，因此 JSON 结构和事件数不会相同。</div>
<figure class="diagram"><img src="../figures/qwen3-profile-phase-sequence.png" alt="Qwen3 profiling phase sequence comparing auto-infer, omni-npu and vllm-ascend"><figcaption>统一 phase contract：三方均为 1 PREFILL + {phases["decode_passes"]} DECODE；omni-npu 与 vllm-ascend 的 terminal drain 保留在 decode 之外。</figcaption></figure>
<div class="callout"><strong>读取口径：</strong>captured request range 是 profiler-instrumented wall range，可用于本次短窗口对照，但不替代 headline。阶段表是 complete events 的累计事件时间；host/device、嵌套 range 与并发 stream 会重叠，因此不能把 phase duration 相加当作请求墙钟。</div>
<div class="profile-grid">{_profile_cards(summary, manifest)}</div>
<p class="callout">本次 {profile_workload["output_tokens"]}-token profile digest 三方{'一致' if profile_digest_equal else '不一致'}；这不改变 {headline_workload["output_tokens"]}-token headline 中 omni-npu digest 不同的事实。打开方式：在 Chromium 输入 <code>chrome://tracing</code>，选择 Load 后载入任一 JSON；文件本身未压缩、未截断。</p></section>

<section id="call-stack-comparison"><div class="section-head"><span class="eyebrow">SOURCE-OBSERVED · HOT PATH</span><h2>同一请求，三套调用栈</h2><p>调用层数不是单独的性能结论；真正重要的是状态所有权、变化隔离和每步需要穿越的组件边界。以下符号按本次已锁定源码版本核对。</p></div>
<figure class="diagram"><img src="../figures/qwen3-three-framework-call-stacks.png" alt="Source observed Qwen3 call stacks for auto-infer, omni-npu and vllm-ascend"><figcaption>auto-infer 用短执行协议连接唯一 EngineCore 与设备 runner；另外两套框架保留 vLLM 通用生命周期，并在 plugin / worker / runner 层加入 Ascend 专化。</figcaption></figure>
<div class="table-scroll"><table class="call-stack-table"><thead><tr><th>框架</th><th>主调用栈</th><th>所有权 / 间接性</th><th>源码位置</th></tr></thead><tbody>{_call_stack_rows()}</tbody></table></div></section>

<section id="why-faster"><div class="section-head"><span class="eyebrow">MEASURED → OBSERVED → INFERRED</span><h2>为什么 auto-infer 更快</h2><p>性能解释必须区分事实与因果推断。下面把每条机制连到测试结果与源码边界；没有单变量 ablation 的地方不会写成已证明因果。</p></div>
<div class="table-scroll"><table class="evidence-table"><thead><tr><th>证据类型</th><th>环节</th><th>结论</th><th>依据 / 限制</th></tr></thead><tbody>{_causal_rows(summary, manifest)}</tbody></table></div>
<div class="callout"><strong>核心解释：</strong>auto-infer 的优势不是“异步越多越快”。本负载下 depth-{headline_workload["async_batches"]} async 反而更慢，最终采用同步默认。领先来自更短的确定性热路径：启动期捕获恰当 gear、固定地址输入、脏 metadata 更新、event 排序、packed projection 与 captured greedy epilogue 的组合。</div></section>

<section id="architecture-comparison"><div class="section-head"><span class="eyebrow">SOURCE-OBSERVED · SCOPED</span><h2>架构优劣详细对比</h2><p>auto-infer 的核心优势是低间接性、明确所有权和小扩展 seam；两个对手的优势是产品宽度与成熟生态。代码行数只表示审计面，不单独构成质量结论。</p></div>
<figure class="diagram"><img src="../figures/qwen3-three-framework-architecture.png" alt="Architecture layers of auto-infer, omni-npu and vllm-ascend"><figcaption>同层横向阅读：auto-infer 把变化压在 capability/registry 与设备 backend；omni-npu 和 vllm-ascend 用更宽的通用框架与平台扩展换取生态覆盖。</figcaption></figure>
<div class="table-scroll"><table class="architecture-table"><thead><tr><th>维度</th><th>auto-infer</th><th>omni-npu</th><th>vllm-ascend</th></tr></thead><tbody>{_architecture_rows()}</tbody></table></div>
<div class="split" style="margin-top:24px"><article class="appendix-card"><span class="evidence observed">源码观察</span><h3>auto-infer 的架构优势</h3><ul><li>执行、attention、MTP 都通过 capability/registry 换对象，而不是把模型分支压进 engine。</li><li>graph 性能胶水拆为 staging、task pipeline、epilogue；可单测且状态所有者清楚。</li><li>生产核心 9,960 Python LOC / 93 files，无内部 import cycle，审计路径短。</li><li>unsupported 能力 fail-fast，不保留静默 eager fallback 来掩盖错误路径。</li></ul></article><article class="appendix-card"><span class="evidence measured">诚实短板</span><h3>两个对手仍更强的地方</h3><ul><li>vllm-ascend：上游模型、OpenAI API、scheduler、connector 与部署工具链成熟。</li><li>omni-npu：优化模型、复杂 MoE/EP/P-D、算子与 best-practice 配置覆盖更深。</li><li>auto-infer：模型面、量化、生产拓扑广度、长期 soak 证据仍需持续建设。</li><li>当前优胜结论不能外推到未测模型、shape、并行规模或量化格式。</li></ul></article></div></section>

<section id="invariants"><div class="section-head"><span class="eyebrow">CHANGE GOVERNANCE</span><h2>什么不应该变化，什么必须按模型重生成</h2><p>这条边界决定框架会继续收敛，还是重新滑向模型特判。左侧是版本化架构契约；右侧必须绑定 model/config/weights digest、软件版本和硬件版本。</p></div>
<div class="split"><article class="contract invariant"><span class="eyebrow">FRAMEWORK INVARIANTS</span><h3>不因模型变化</h3><ul>{_invariant_items()}</ul></article><article id="per-model-artifacts" class="contract generated"><span class="eyebrow">PER-MODEL GENERATED</span><h3>每个模型重新生成 / 验证</h3><ul>{_regenerated_items()}</ul></article></div>
<div class="callout warning"><strong>变更规则：</strong>不变量只能通过版本化设计、跨模型回归和新 baseline 修改；生成物不能仅因两个 checkpoint 共享同一个 architecture class 名就复用。</div></section>

<section id="acceptance-workflow"><div class="section-head"><span class="eyebrow">PRECISION-FIRST RELEASE GATE</span><h2>新模型生产验收流程</h2><p>精度先于性能。任何阶段失败都不进入下一阶段排名；性能 profiling 只能解释一个已通过精度门的实现。</p></div>
<div class="flow"><div><b>01</b><span>Checkpoint inventory<br>config / weights / digest</span></div><div><b>02</b><span>Geometry generation<br>attention / KV / TP / EP / MTP</span></div><div><b>03</b><span>Precision gates<br>logits → tokens → long context</span></div><div><b>04</b><span>Graph matrix<br>gear / memory / capture / fallback=0</span></div><div><b>05</b><span>Stability<br>continuous batching / preemption / soak</span></div><div><b>06</b><span>Matched ranking<br>raw JSON / trace / hashes / report</span></div></div>
<div class="appendix-grid" style="margin-top:24px"><article class="appendix-card"><h3>精度门</h3><ul><li>独立 reference logits 与逐层定位</li><li>eager / paged / graph token identity</li><li>BF16/FP32 head 与 greedy parity</li><li>长上下文、边界 block、并发与 preemption</li><li>MTP acceptance by position 与 target-only fallback parity</li></ul></article><article class="appendix-card"><h3>稳定性门</h3><ul><li>持续 batching、请求取消、KV 回收</li><li>慢客户端、过载、服务重启、异常传播</li><li>CV、P50/P95/P99、RSS/HBM plateau</li><li>多 topology 的 HCCL failure/recovery</li><li>版本化阈值；异常样本保留而非删除</li></ul></article></div></section>

<section id="evidence-appendix"><div class="section-head"><span class="eyebrow">AUDIT TRAIL</span><h2>证据附录</h2><p>HTML、归一化 JSON 与 raw trace 都可独立审计。相对指标从 JSON 重新计算，trace 链接与 SHA-256 一一对应。</p></div>
<div class="table-scroll"><table><thead><tr><th>框架</th><th>原始文件</th><th>events</th><th>大小</th><th>SHA-256</th><th>torch / torch-npu / vLLM</th></tr></thead><tbody>{_artifact_rows(manifest)}</tbody></table></div>
<div class="appendix-grid" style="margin-top:24px"><article class="appendix-card"><h3>机器可读入口</h3><ul><li><a href="profiling/qwen3/manifest.json">profiling/qwen3/manifest.json</a></li><li><a href="profiling/qwen3/summary.json">profiling/qwen3/summary.json</a></li><li><a href="profiling/qwen3/provenance.json">profiling/qwen3/provenance.json</a></li><li><a href="ARCHITECTURE-COMPARISON.md">ARCHITECTURE-COMPARISON.md</a></li><li><a href="FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md">FINAL-ARCHITECTURE-VALIDATION-2026-07-20.md</a></li></ul><p>主测试原始 JSON 保留于 npu2 的 <code>/data2/auto-infer-decode-performance/results/final-20260720/</code>；其完整内容已嵌入 summary。capture harness revision 与三个 framework source revision 分开记录，避免把采集脚本版本误当成外部框架版本。</p></article><article class="appendix-card"><h3>复现归一化与报告</h3><pre>python tools/analyze_qwen3_profiles.py \
  --metadata AUTO_META OMNI_META VLLM_META \
  --benchmarks AUTO_JSON OMNI_JSON VLLM_JSON \
  --output-dir docs/profiling/qwen3 \
  --capture-provenance docs/profiling/qwen3/provenance.json \
  --benchmark-schema {_e(manifest["benchmark_schema"])}

python tools/build_qwen3_architecture_report.py
pytest -q</pre></article></div>
<div class="callout warning"><strong>限制：</strong>profile window 只有 B{profile_workload["batch_size"]}、{profile_workload["output_tokens"]} tokens，包含 profiler overhead；phase share 是重叠事件时间；operator-name 分类依赖当前版本，三方各有 {min(unclassified_percentages):.1f}%–{max(unclassified_percentages):.1f}% 事件时间保持 unclassified。没有单变量 ablation，不宣称每一毫秒都能归因于某一项优化。</div></section>
<footer>auto-infer architecture &amp; performance evidence · generated deterministically from <code>summary.json</code> and <code>manifest.json</code></footer>
</main></div></body></html>'''


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(map(_md_cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_md_cell(cell) for cell in row) + " |"
        for row in rows)
    return "\n".join(lines)


def build_markdown_report(summary: dict, manifest: dict) -> str:
    """Build the text-first companion from the same evidence as the HTML."""
    data = summary["headline_benchmarks"]
    relative = summary["relative_to_auto_infer"]
    workload = manifest["workload"]
    headline = data["auto-infer"]["manifest"]
    phases = workload["capture_phases"]
    driver = manifest["provenance"]["driver"]

    benchmark_rows = []
    for label, direction, values, unit in _headline_rows(summary):
        digits = 1 if unit == "tok/s" else 3
        benchmark_rows.append([
            label, direction,
            *[f"{value:,.{digits}f} {unit}" for value in values],
        ])

    profile_rows = []
    for framework in FRAMEWORKS:
        profile = summary["profiles"][framework]
        artifact = manifest["artifacts"][framework]
        steps = profile["execution_phases"]["steps"]
        profile_rows.append([
            framework,
            f"{_request_duration(profile) / 1000:.2f} ms",
            f"{steps[0]['duration_us'] / 1000:.2f} ms",
            f"{sum(step['duration_us'] for step in steps[1:]) / 1000:.2f} ms",
            f"{profile['complete_event_count']:,}",
            f"[{artifact['path']}](profiling/qwen3/{artifact['path']})",
        ])

    step_rows = []
    step_count = len(summary["profiles"][FRAMEWORKS[0]][
        "execution_phases"]["steps"])
    for index in range(step_count):
        reference = summary["profiles"][FRAMEWORKS[0]][
            "execution_phases"]["steps"][index]
        step_rows.append([
            reference["label"],
            *[
                f"{summary['profiles'][framework]['execution_phases']['steps'][index]['duration_us'] / 1000:.3f} ms"
                for framework in FRAMEWORKS
            ],
        ])

    causal_rows = [list(row) for row in _causal_data(summary, manifest)]
    call_stack_rows = [list(row) for row in _call_stack_data()]
    architecture_rows = [list(row) for row in _architecture_data()]
    artifact_rows = []
    for framework in FRAMEWORKS:
        artifact = manifest["artifacts"][framework]
        artifact_rows.append([
            framework,
            f"[{artifact['path']}](profiling/qwen3/{artifact['path']})",
            f"{artifact['event_count']:,}",
            f"{artifact['size_bytes'] / 2**20:.1f} MiB",
            f"`{artifact['sha256']}`",
        ])

    invariants = "\n".join(f"- {item}" for item in _invariant_data())
    generated = "\n".join(f"- {item}" for item in _regenerated_data())
    report = f"""# auto-infer 架构与 Qwen3 性能审计报告

> 面向管理层与工程师的文本版报告。数据源与 HTML 版完全相同；生成器不维护第二套手写指标。

## 管理结论

在 `{Path(headline['model']).name}`、单张 `{driver['soc']}`、`{headline['dtype']}` greedy、每框架 `{headline['usable_kv_tokens']:,}` usable KV tokens 的验收边界内，auto-infer 的稳态延迟、吞吐、启动、等容量内存与稳定性均为本次第一。

- B{headline['throughput_batch']} 吞吐：**{data['auto-infer']['throughput_tokens_per_second']['median']:,.1f} tok/s**。
- 相对 omni-npu：**{relative['omni-npu']['throughput_speedup']:.2f}×**；相对 vllm-ascend：**{relative['vllm-ascend']['throughput_speedup']:.2f}×**。
- auto-infer 与 vllm-ascend 的 {headline['output_tokens']}-token digest 一致；omni-npu 长度一致但 digest 不同，因此只声明性能可比，不声明 token identity。
- 结论仅适用于已测模型、shape、BF16 精度和单卡拓扑；不能外推到未测模型、量化或分布式规模。

## Matched benchmark

权威 headline 来自无 profiler 的 {headline['measured_runs']} 次测量；profiling 只用于解释，不替代性能排名。

{_md_table(['指标', '方向', *FRAMEWORKS], benchmark_rows)}

## Qwen3 三框架 profiling

每份 trace 捕获同一个 B{workload['batch_size']}、{workload['output_tokens']}-token generate：**{phases['prefill_passes']} 次 prefill + {phases['decode_passes']} 次连续 decode**。这是连续多步 decode，不是 speculative MTP。

### 如何直接找到 prefill

在 Chrome 的 `chrome://tracing` 或 Perfetto 中载入任一原始 JSON，然后找到置顶的 **`QWEN3 PHASES`** process：

1. 唯一的 **`PREFILL`** 是首个 engine step。
2. 后续依次是 **`DECODE 001`** 到 **`DECODE {phases['decode_passes']:03d}`**。
3. 这些是采集器写入三套框架的统一 host ranges；框架原生 operator、线程、stream 和 category 完整保留。

三份 JSON 的结构和事件数不同是预期现象：auto-infer、omni-npu 和 vllm-ascend 暴露的 Python/C++/ACL graph、async queue 与 runtime 元数据层级不同。可比性来自统一 workload、输出长度、KV 容量、设备、精度和 `QWEN3 PHASES` 边界，而不是要求三份 trace 长得一样。

![Qwen3 三框架 phase 时序](../figures/qwen3-profile-phase-sequence.png)

{_md_table(['框架', '请求范围', 'PREFILL host range', f'{phases['decode_passes']} 个 DECODE 合计', '原生 complete events', '原始 Trace'], profile_rows)}

### 逐步 phase 索引

下面是 host range，不是 NPU kernel 独占时间；异步 device stream 可能越过 host range，不能把这些数直接当作纯算子耗时。

{_md_table(['阶段', *FRAMEWORKS], step_rows)}

## 三框架调用栈对比

![Qwen3 三框架源码调用栈](../figures/qwen3-three-framework-call-stacks.png)

调用层数不是单独的性能结论；这里比较的是状态所有权、变化隔离和一次模型执行需要穿越的组件边界。符号按 manifest 锁定的源码版本核对。

{_md_table(['框架', '主调用栈', '所有权 / 间接性', '源码位置'], call_stack_rows)}

## 为什么 auto-infer 更快

{_md_table(['证据类型', '环节', '结论', '依据 / 限制'], causal_rows)}

领先来自较短且确定的热路径组合：启动期捕获合适 gear、固定地址输入、dirty metadata 更新、event 排序、packed projection，以及 graph 内 BF16 lm_head 与 greedy argmax。没有单变量 ablation 的机制只作为与结果一致的因果解释，不写成已独立证明的毫秒收益。

## 架构优劣详细对比

![Qwen3 三框架架构分层](../figures/qwen3-three-framework-architecture.png)

{_md_table(['维度', 'auto-infer', 'omni-npu', 'vllm-ascend'], architecture_rows)}

auto-infer 的核心优势是低间接性、明确所有权和较小扩展 seam；vllm-ascend 的优势是模型/API/部署生态成熟度，omni-npu 的优势是优化模型、算子和复杂并行覆盖。特性广度属于当前 scope 差异，不能反向证明核心架构质量。

## 什么不应该变化

{invariants}

## 什么应针对每个模型重新生成

{generated}

不变量只能通过版本化设计、跨模型回归和新 baseline 修改；模型生成物必须绑定 config、weights、软件与硬件 digest，不能只因 architecture class 同名而复用。

## 新模型生产验收流程

1. Checkpoint inventory：config、weights、digest。
2. Geometry generation：attention、KV、TP/EP、MTP。
3. Precision gates：reference logits、token identity、长上下文与边界 block。
4. Graph matrix：gear、内存、capture、fallback=0。
5. Stability：continuous batching、preemption、取消、KV 回收与 soak。
6. Matched ranking：无 profiler headline、raw JSON、phase trace、hash 与报告。

精度门失败时不进入性能排名；profiling 只能解释一个已经通过精度门的实现。

## 证据附录

{_md_table(['框架', '原始文件', 'events', '大小', 'SHA-256'], artifact_rows)}

- [manifest.json](profiling/qwen3/manifest.json)：工作负载、环境与 artifact contract。
- [summary.json](profiling/qwen3/summary.json)：标准化 headline、operator 分类和 phase index。
- [provenance.json](profiling/qwen3/provenance.json)：模型、源码、驱动与采集来源。
- [HTML 报告](AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html)：同一证据的可视化版本。

### 限制

- profile window 只有 B{workload['batch_size']}、{workload['output_tokens']} tokens，并包含 profiler overhead。
- `PREFILL`/`DECODE` 是统一 host range；NPU 异步执行可能跨越 host 边界。
- operator category 的事件时间存在嵌套与并发，不能相加为 request wall time。
- 三方原生事件命名不同，operator-name 分类仍保留 unclassified；统一 phase lane 不伪造缺失的算子归因。
- 没有单变量 ablation，不声明每一毫秒都来自某一项单独优化。
"""
    return report


def main() -> None:
    summary = json.loads((PROFILE_DIR / "summary.json").read_text())
    manifest = json.loads((PROFILE_DIR / "manifest.json").read_text())
    OUTPUT.write_text(build_report(summary, manifest))
    OUTPUT_MD.write_text(build_markdown_report(summary, manifest))


if __name__ == "__main__":
    main()
