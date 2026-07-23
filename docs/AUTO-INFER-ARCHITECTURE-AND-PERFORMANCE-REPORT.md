# auto-infer 架构与 Qwen3 性能审计报告

> 面向管理层与工程师的文本版报告。数据源与 HTML 版完全相同；生成器不维护第二套手写指标。

## 管理结论

在 `Qwen3-0.6B`、单张 `Ascend 910B1`、`bfloat16` greedy、每框架 `14,464` usable KV tokens 的验收边界内，auto-infer 的稳态延迟、吞吐、启动与稳定性均为本次第一。内存是明确代价：peak torch allocation 为 5.24 GiB，高于 vllm-ascend 的 2.80 GiB，但低于 omni-npu 的 9.73 GiB。

- B16 吞吐：**2,380.4 tok/s**。
- 相对 omni-npu：**1.23×**；相对 vllm-ascend：**2.72×**。
- auto-infer 与 vllm-ascend 的 128-token digest 一致；omni-npu 长度一致但 digest 不同，因此只声明性能可比，不声明 token identity。
- 结论仅适用于已测模型、shape、BF16 精度和单卡拓扑；不能外推到未测模型、量化或分布式规模。

## Matched benchmark

权威 headline 来自无 profiler 的 20 次测量；profiling 只用于解释，不替代性能排名。

| 指标 | 方向 | auto-infer | omni-npu | vllm-ascend |
| --- | --- | --- | --- | --- |
| Warm TTFT | 越低越好 | 6.137 ms | 55.105 ms | 18.344 ms |
| TPOT | 越低越好 | 5.672 ms | 6.710 ms | 17.405 ms |
| B16 throughput | 越高越好 | 2,380.4 tok/s | 1,930.2 tok/s | 874.4 tok/s |
| Engine load + graph | 越低越好 | 8.435 s | 34.907 s | 44.858 s |
| Peak torch allocation | 越低越好 | 5.239 GiB | 9.731 GiB | 2.797 GiB |
| Throughput CV | 越低越好 | 1.348 % | 1.702 % | 1.813 % |

## Qwen3 三框架 profiling

每份 trace 捕获同一个 B16、16-token generate：**1 次 prefill + 15 次连续 decode**。这是连续多步 decode，不是 speculative MTP。

**Prefill 执行模式门禁：**B16 × 9-token prompt 形成 144 个 flattened query tokens。prefill 使用独立的 256-token graph cap，不与 decode request gear 共用上限；采集器与发布 analyzer 都强制要求 `prefill-graph=1`、`eager=0`、`online capture=0`，否则拒绝发布 trace。

### 如何直接找到 prefill

在 Chrome 的 `chrome://tracing` 或 Perfetto 中载入任一原始 JSON，然后找到置顶的 **`QWEN3 PHASES`** process：

1. 唯一的 **`PREFILL`** 是首个 engine step。
2. 后续依次是 **`DECODE 001`** 到 **`DECODE 015`**。
3. 这些是采集器写入三套框架的统一 host ranges；框架原生 operator、线程、stream 和 category 完整保留。

三份 JSON 的结构和事件数不同是预期现象：auto-infer、omni-npu 和 vllm-ascend 暴露的 Python/C++/ACL graph、async queue 与 runtime 元数据层级不同。可比性来自统一 workload、输出长度、KV 容量、设备、精度和 `QWEN3 PHASES` 边界，而不是要求三份 trace 长得一样。

![Qwen3 三框架 phase 时序](../figures/qwen3-profile-phase-sequence.png)

| 框架 | 请求范围 | PREFILL host range | 15 个 DECODE 合计 | 原生 complete events | 原始 Trace |
| --- | --- | --- | --- | --- | --- |
| auto-infer | 131.52 ms | 11.73 ms | 118.73 ms | 53,801 | [raw/auto-infer.trace.json](profiling/qwen3/raw/auto-infer.trace.json) |
| omni-npu | 204.32 ms | 57.85 ms | 140.46 ms | 41,301 | [raw/omni-npu.trace.json](profiling/qwen3/raw/omni-npu.trace.json) |
| vllm-ascend | 363.56 ms | 22.06 ms | 334.18 ms | 38,838 | [raw/vllm-ascend.trace.json](profiling/qwen3/raw/vllm-ascend.trace.json) |

### 逐步 phase 索引

下面是 host range，不是 NPU kernel 独占时间；异步 device stream 可能越过 host range，不能把这些数直接当作纯算子耗时。

| 阶段 | auto-infer | omni-npu | vllm-ascend |
| --- | --- | --- | --- |
| PREFILL | 11.727 ms | 57.850 ms | 22.058 ms |
| DECODE 001 | 7.995 ms | 9.207 ms | 22.888 ms |
| DECODE 002 | 7.644 ms | 11.975 ms | 21.935 ms |
| DECODE 003 | 7.630 ms | 9.229 ms | 21.747 ms |
| DECODE 004 | 7.574 ms | 9.148 ms | 22.722 ms |
| DECODE 005 | 7.657 ms | 9.058 ms | 21.785 ms |
| DECODE 006 | 7.763 ms | 9.049 ms | 21.724 ms |
| DECODE 007 | 7.677 ms | 9.010 ms | 22.094 ms |
| DECODE 008 | 8.142 ms | 9.310 ms | 22.294 ms |
| DECODE 009 | 7.963 ms | 9.101 ms | 22.231 ms |
| DECODE 010 | 8.647 ms | 9.197 ms | 23.240 ms |
| DECODE 011 | 8.127 ms | 9.244 ms | 22.296 ms |
| DECODE 012 | 8.025 ms | 9.192 ms | 22.277 ms |
| DECODE 013 | 7.885 ms | 9.174 ms | 22.301 ms |
| DECODE 014 | 7.919 ms | 9.267 ms | 22.283 ms |
| DECODE 015 | 8.081 ms | 9.297 ms | 22.364 ms |

## Trace 里的真实调用栈对比

**TRACE-DERIVED · MEASURED HOST RANGES**

![Qwen3 三框架 Trace 实测调用栈](../figures/qwen3-trace-call-stack-comparison.svg)

每个 bar 都来自运行时真正被调用的 `qwen3/call` profiler range。Decode 图选择 15 步中最接近中位数的实际 step，不挑最优样本。在任一原始 JSON 中找到置顶的 **`QWEN3 CALL STACK`** process，即可核对相同嵌套 range。

| 框架 | PREFILL | 15-step decode 中位数 | 最大嵌套 | Trace 实测边界（中位 step） |
| --- | --- | --- | --- | --- |
| auto-infer | 11.73 ms (baseline) | 7.92 ms (baseline) | 5 | d0 EngineCore.step · d1 Scheduler.schedule · d1 GraphPagedNpuExecutor.execute · d2 GraphPagedRunner.execute · d3 GraphPagedRunner.submit · d4 GraphPagedRunner._graph_submit |
| omni-npu | 57.85 ms (4.93× slower than auto-infer) | 9.20 ms (1.16× slower than auto-infer) | 7 | d0 LLMEngine.step · d1 InprocClient.get_output · d2 EngineCore.step_fn · d3 AsyncScheduler.schedule · d3 UniProcExecutor.execute_model · d4 WorkerWrapperBase.execute_model · d5 NPUWorker.execute_model · d6 NPUModelRunner.execute_model |
| vllm-ascend | 22.06 ms (1.88× slower than auto-infer) | 22.28 ms (2.81× slower than auto-infer) | 7 | d0 LLMEngine.step · d1 InprocClient.get_output · d2 EngineCore.step_fn · d3 AsyncScheduler.schedule · d3 UniProcExecutor.execute_model · d4 WorkerWrapperBase.execute_model · d5 NPUWorker.execute_model · d6 NPUModelRunner.execute_model |

Trace 证明本次 auto-infer 的 prefill 与 decode host range 都更短，且 prefill 实际经过 `GraphPagedRunner._prefill_graph_submit`。这不等于证明所有未插桩函数都不存在，也不能单独证明“层数更少”就是全部加速因果。`record_function` 对每个边界也有小幅且不同数量的扰动，排名仍以无 profiler headline 为准。

## 为什么 auto-infer 更快

| 证据类型 | 环节 | 结论 | 依据 / 限制 |
| --- | --- | --- | --- |
| 实测 | B16 throughput | 2,380.4 tok/s；较 omni-npu 1.23×，较 vllm-ascend 2.72×。 | headline benchmark JSON |
| 源码观察 | Graph hot path | graph capture、staging、replay/update、epilogue 拆成可独立测试组件；热路径不做在线 capture。 | graph_decode_runner / graph_task_pipeline |
| 因果推断 | Replay + metadata pipeline | replay 后 side-stream 更新、event 排序和双缓冲减少 graph-task update 对下一步的阻塞。 | 与低 TPOT 及短 profiled request 一致；未做单变量 ablation |
| 源码观察 | Persistent staging | CPU/NPU 输入缓冲持久化，block table 仅上传 dirty row/span。 | staging / input stagers |
| 因果推断 | 较少 host/device 胶水 | 固定地址与脏更新降低逐步分配、拷贝和 Python 调度成本。 | trace 中 auto-infer request range 最短 |
| 源码观察 | Packed projections | QKV 与 gate/up 使用 packed weight；BF16 lm_head 与 greedy argmax 留在 captured epilogue。 | packed projections / decode epilogue |
| 因果推断 | 更少 kernel 与同步边界 | projection packing 与直接 argmax 降低 launch 数；收益随模型/shape 变化，必须重做 profiling。 | 机制合理但不能由相关性证明全部增益 |
| 实测 | Profiler window | B16 16-token 请求范围约 131.5 / 204.3 / 363.6 ms（auto / omni / vllm）。 | 三份 raw Chrome Trace |
| 实测 | Startup | 8.435 s vs 34.907 s vs 44.858 s；auto-infer 与 vllm-ascend 都覆盖到 256-token graph family，Omni 覆盖到 512；启动差距不能只归因于 gear 数。 | headline benchmark + captured framework configuration logs |

领先来自较短且确定的热路径组合：启动期捕获合适 gear、固定地址输入、dirty metadata 更新、event 排序、packed projection，以及 graph 内 BF16 lm_head 与 greedy argmax。没有单变量 ablation 的机制只作为与结果一致的因果解释，不写成已独立证明的毫秒收益。

## 架构优劣详细对比

![Qwen3 三框架架构分层](../figures/qwen3-three-framework-architecture.png)

| 维度 | auto-infer | omni-npu | vllm-ascend |
| --- | --- | --- | --- |
| 核心控制流 | EngineCore → BatchPlan → Executor → ExecutionResult；协议短且状态归属明确。 | vLLM 主流程之上叠加环境选择的 patch 与额外配置。 | 复用成熟 vLLM engine，Ascend worker / runner / compiler 专化。 |
| 模型扩展 seam | 模型声明 attention / MTP capability；registry 注入对象，engine 不按模型分支。 | 模型实现、best-practice config 与 patches 共同决定路径。 | 覆盖广，但模型与平台 runner 的组合触点更多。 |
| 状态所有权 | Engine/service 单线程拥有 request、scheduler、KV、completion；跨线程传不可变视图。 | 上游 vLLM 状态与 patch 后行为共同形成所有权边界。 | 继承 vLLM 的多组件生命周期，成熟但阅读跨度更大。 |
| Graph 生命周期 | 启动期捕获；固定地址 staging；replay 后独立 stream metadata update；event + 双缓冲。 | NpuGraphEx + full/piecewise graph，依赖插件配置与 graph gear。 | torch.compile + ACL graph piecewise；支持较宽的通用 shape 集合。 |
| 输入与 KV metadata | 持久 pinned CPU/NPU buffer；block table 只传 dirty rows/span。 | 由 Omni runner 与 patched vLLM metadata 路径管理。 | 通用 input batch 和 worker metadata 路径，支持面广。 |
| Projection / epilogue | packed QKV、gate/up；BF16 captured lm_head + greedy argmax；避免外部 sampler step。 | 拥有广泛融合算子与模型专用优化配置。 | 平台 custom op、compiler fusion 与通用 sampler 体系。 |
| Continuous batching | scheduler/KV 生命周期有专项回归；同步路径是本负载已验证默认。 | 继承 vLLM scheduling，并通过 patch 增补行为。 | vLLM V1 scheduler 成熟，线上生态更完整。 |
| Serving | 单 service + broker + request-id demux；在线/离线共用核心执行协议。 | 成熟 vLLM serving，Omni patch OpenAI 与 scheduler 层。 | API、工具链、部署经验最成熟。 |
| Distributed / MoE | 命名 TP/DP/EP/CP/SP mesh；BF16 true all-to-all 接口与测试；深度仍有限。 | 并行与算子调优面最丰富，是明显强项。 | 上游并行体系成熟，Ascend 通信优化覆盖更广。 |
| MTP | 一个 two-stage recurrent graph path；geometry 从权重推导；unsupported fail-fast。 | Eagle / MTP patches 与模型优化覆盖更广。 | 上游 speculative decoding 生态更完整。 |
| P/D 与 MLA MTP | 仅保留未接线 P/D 低层接口；MLA MTP capability 保留但明确 unsupported。 | P/D、connector 与 MLA/MoE 产品能力更完整。 | connector / disaggregation 生态成熟。 |
| 维护与审计面 | 9,960 Python LOC / 93 files；无内部 import cycle；路径较短。 | 61,080 / 223；patch 提升适配力，也增加组合状态。 | 53,219 / 242；生态收益大，平台 runner 体量更高。 |

auto-infer 的核心优势是低间接性、明确所有权和较小扩展 seam；vllm-ascend 的优势是模型/API/部署生态成熟度，omni-npu 的优势是优化模型、算子和复杂并行覆盖。特性广度属于当前 scope 差异，不能反向证明核心架构质量。

## 什么不应该变化

- EngineCore → BatchPlan → Executor → ExecutionResult 协议
- request / scheduler / KV / completion 的单一所有权
- 模型声明 capability、registry 选择实现；engine 不加模型分支
- recurrent MTP 独立 capability；不支持时启动期 fail-fast
- graph-FIA capture / replay / update 的 event 顺序契约
- 固定地址持久 staging 与 dirty block-table 更新
- 精度优先：logits/token parity 先于性能排名
- P/D 未接线、MLA MTP unsupported 等边界必须显式
- matched manifest、raw samples、trace 和 hash 的证据保留

## 什么应针对每个模型重新生成

- checkpoint / config / weight-name inventory 与 adapter
- TP/EP head、expert、attention、RoPE 与 cache geometry
- KV budget、block size、scratch blocks、max sequence length
- packed QKV / gate-up 权重与 dtype / quantization metadata
- graph gear ladder、capture matrix、handles、memory envelope
- MTP layers、geometry、depth、position acceptance 与 capability
- BF16/FP32 head 与 sampling parity 阈值
- golden prompts、logits/token digest、eager/paged/graph identity
- unprofiled baseline、raw traces、phase map、CV 与回归阈值
- 实际声明的 TP/EP/SP/CP 拓扑验证

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

| 框架 | 原始文件 | events | 大小 | SHA-256 |
| --- | --- | --- | --- | --- |
| auto-infer | [raw/auto-infer.trace.json](profiling/qwen3/raw/auto-infer.trace.json) | 70,882 | 11.3 MiB | `1bff59c92323b26b2ae7da442d29027a560c1853a2001ce5f53a2ac264e119fa` |
| omni-npu | [raw/omni-npu.trace.json](profiling/qwen3/raw/omni-npu.trace.json) | 60,039 | 9.2 MiB | `2c36a2e2f55cd7b4f112de22eea83a15477bfa4e673d7d7d6bd7fc843067e1a7` |
| vllm-ascend | [raw/vllm-ascend.trace.json](profiling/qwen3/raw/vllm-ascend.trace.json) | 55,663 | 8.4 MiB | `808c99f6410647485bec89933cb02de670203725de01ea43dcadd8febede6fa4` |

- [manifest.json](profiling/qwen3/manifest.json)：工作负载、环境与 artifact contract。
- [summary.json](profiling/qwen3/summary.json)：标准化 headline、operator 分类和 phase index。
- [provenance.json](profiling/qwen3/provenance.json)：模型、源码、驱动与采集来源。
- [HTML 报告](AUTO-INFER-ARCHITECTURE-AND-PERFORMANCE-REPORT.html)：同一证据的可视化版本。

### 限制

- profile window 只有 B16、16 tokens，并包含 profiler overhead。
- `PREFILL`/`DECODE` 是统一 host range；NPU 异步执行可能跨越 host 边界。
- operator category 的事件时间存在嵌套与并发，不能相加为 request wall time。
- 三方原生事件命名不同，operator-name 分类仍保留 unclassified；统一 phase lane 不伪造缺失的算子归因。
- 没有单变量 ablation，不声明每一毫秒都来自某一项单独优化。
