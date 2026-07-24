# Qwen2.5-72B TP8 三框架在线推理报告

日期：2026-07-24

设备：单机 8 × Ascend 910B1

模型：`Qwen2.5-72B-Instruct`

权重：BF16

接口：OpenAI streaming completion，greedy，`ignore_eos=true`

## 管理结论

在本次同机、同模型、同请求的 B1/B4/B16 测试中，auto-infer 的聚合输出吞吐全部第一：

| 并发 | auto-infer | Omni-NPU | vllm-ascend | 相对第二名 |
|---|---:|---:|---:|---:|
| B1 | **35.68 tok/s** | 31.64 tok/s | 20.25 tok/s | **+12.8%** |
| B4 | **124.57 tok/s** | 111.71 tok/s | 73.23 tok/s | **+11.5%** |
| B16 | **394.77 tok/s** | 339.27 tok/s | 260.40 tok/s | **+16.4%** |

其中 B1、B4、B16 分别表示同时保持 1、4、16 个并发请求。每个数字是三次独立测量的均值，不是挑选最好的一次。三框架所有测量均为 0 failed、0 rejected。

auto-infer 的优势不是降低精度换来的：同一服务重复运行的单请求、B16 连续批处理和 prefix-cache 输出均做到 token 级完全一致；prefix cache 的命中统计也正确显示 45/60 blocks、75% hit rate。

需要诚实保留一个非第一项：B4 TTFT 为 216.3 ms，慢于 vllm-ascend 的 158.1 ms，但快于 Omni-NPU 的 326.2 ms。吞吐、ITL，以及 B1/B16 TTFT 均领先。

## 测试口径

- checkpoint：`/data1/models/Qwen2.5-72B-Instruct`
- tensor parallel：TP8
- prompt：`Explain tensor parallel inference performance with clear evidence.`
- tokenizer 实际输入：9 tokens
- 每请求输出：64 tokens
- workload：B1 共 8 个请求，B4 共 20 个请求，B16 共 32 个请求
- 每组先做 2 个 warm-up 请求
- 每个框架、每个并发独立运行 3 次
- 吞吐：完成的输出 token 数 / workload wall time
- TTFT、ITL、E2E：每次运行中所有 streaming 请求的 p50，再对 3 次运行取均值
- auto-infer：graph mode，`max_model_len=4096`、`num_blocks=2048`、`max_num_seqs=32`、`max_gear=16`
- vllm-ascend：vLLM 0.20.2 + vllm-ascend 0.20.2rc1
- Omni-NPU：vLLM 0.14.0 + omni-npu 0.2.0

该短 prompt workload 主要衡量在线 decode 和小 prefill；它不能替代 1K/4K prompt、KV 容量和多租户长稳测试。

## 完整性能结果

### 聚合输出吞吐

单位：tok/s，越高越好。

| 并发 | auto-infer 三次结果 | 均值 ± 标准差 | CV | Omni-NPU 均值 | vllm-ascend 均值 |
|---|---|---:|---:|---:|---:|
| B1 | 35.664 / 35.661 / 35.725 | **35.683 ± 0.036** | 0.10% | 31.643 | 20.251 |
| B4 | 122.989 / 125.501 / 125.214 | **124.568 ± 1.375** | 1.10% | 111.707 | 73.228 |
| B16 | 390.957 / 397.625 / 395.738 | **394.773 ± 3.437** | 0.87% | 339.269 | 260.403 |

CV 是变异系数，即标准差除以均值。它把波动换算成相对百分比；越低表示重复运行越稳定。auto-infer 三档 CV 均不超过 1.10%。

相对 vllm-ascend，auto-infer 在 B1/B4/B16 分别领先 76.2%、70.1%、51.6%；相对 Omni-NPU 分别领先 12.8%、11.5%、16.4%。

### TTFT p50

单位：ms，越低越好。TTFT 包括本次真实在线链路的 admission、调度、prefill 和首 token 返回。

| 并发 | auto-infer | Omni-NPU | vllm-ascend | 第一名 |
|---|---:|---:|---:|---|
| B1 | **52.5** | 165.1 | 64.5 | auto-infer |
| B4 | 216.3 | 326.2 | **158.1** | vllm-ascend |
| B16 | **255.0** | 421.4 | 336.3 | auto-infer |

B4 的 10 ms idle admission window 会等待同波请求到齐，稳定了首次 batch shape，但也会放大部分请求的排队时间。这是精度/形状稳定性优先的明确取舍，后续应做自适应 admission，而不是取消确定性门禁。

### ITL p50

单位：ms，越低越好。ITL 最直接反映稳态 decode 每 token 的关键路径。

| 并发 | auto-infer | Omni-NPU | vllm-ascend | 第一名 |
|---|---:|---:|---:|---|
| B1 | **27.50** | 29.27 | 48.88 | auto-infer |
| B4 | **28.94** | 30.72 | 52.61 | auto-infer |
| B16 | **36.95** | 38.07 | 53.62 | auto-infer |

### E2E p50

单位：s，越低越好。

| 并发 | auto-infer | Omni-NPU | vllm-ascend |
|---|---:|---:|---:|
| B1 | **1.794** | 2.022 | 3.157 |
| B4 | **2.044** | 2.280 | 3.487 |
| B16 | **2.589** | 2.981 | 3.886 |

## 为什么 auto-infer 更快

### 1. TP decode 是整模型 graph replay，不是 80 层逐步 host dispatch

Qwen2.5-72B 有 80 层。auto-infer 对常用 batch gear 预捕获完整 decode 图，每一步只更新持久化 metadata 并 replay。lm_head 和稳定 greedy argmax 位于设备路径，避免每 token 重复构图、创建 tensor 或走 softmax/multinomial。

这直接体现在 ITL：B16 为 36.95 ms，低于 Omni-NPU 的 38.07 ms 和 vllm-ascend 的 53.62 ms。

### 2. TP prefill graph 消除了首轮 eager 冷路径

此前 72B 路径只对 decode 使用 graph，prefill 仍 eager，B1 TTFT 约 200 ms。现在每个 TP rank 在启动时以 barrier → capture → barrier 的顺序预热同一组 prefill gear，避免 rank 间捕获顺序漂移。B1 TTFT 降到 52.5 ms。

TP 模式只捕获到 `max_gear`，不会按默认 `max_prefill_tokens=256` 创建 35 张 72B 全模型图；更大的 prefill 保留正确的 eager fallback。这个边界同时控制启动成本、HBM 和代码复杂度。

### 3. 连续批处理的形状和通信结果可复现

首波并发请求如果被拆成 2+14、3+13 等不同 admission shape，BF16 的 collective 次序和近似并列 logits 可能造成 greedy 分叉。auto-infer 在空闲副本上用 10 ms window 收拢同一波请求，并在 graph TP worker 中固定开启 HCCL/LCCL deterministic。

这不是用串行执行规避问题：请求仍进入同一个 continuous batch；B16 吞吐反而达到 394.77 tok/s。

### 4. 更窄、更直接的热路径

核心执行链保持为：

`Serving → EngineCore/Scheduler → BatchPlan → GraphPagedRunner → Model.forward(ctx) → TP graph/HCCL`

同一 `model.forward(ctx)` 同时服务 eager、paged 和 graph；attention backend 通过 registry 注入。TP graph 没有复制模型数学，也没有依赖 vLLM plugin/monkeypatch。性能策略集中在 runner、staging 和 collective 边界，模型文件只声明模型能力。

相较之下，vllm-ascend 的优势是模型与生产特性覆盖广、生态成熟，但插件层、vLLM 状态机和多种 graph 模式增加了热路径间接性；Omni-NPU 的专用 NPU patch/fusion 很强，但 patch 契约和上游版本耦合更重。auto-infer 当前用更小的执行面换来了 Qwen2.5 BF16 TP8 的可控性和更低 ITL。

## 正确性与稳定性门禁

本轮验证文件记录：

- 单请求两次运行：32/32 token 完全一致
- B16 continuous batching：16/16 请求、每请求 32/32 token 完全一致
- prefix cache 重复请求：token 完全一致
- prefix cache metrics：queried 60 blocks，hit 45 blocks，hit rate 75%
- 性能测试：三个框架合计 27 个独立 run，全部 0 failed、0 rejected
- auto-infer 吞吐 CV：0.10% / 1.10% / 0.87%
- 本地代码回归：591 tests passed

这里的“精度通过”指同 checkpoint、同服务路径的确定性 token parity 和 cache/continuous-batching 一致性。正式生产发布仍应补充 reference logits、长上下文 corpus、任务精度集和 24–72 小时故障注入；本报告不把自一致性夸大为跨框架 bitwise 等价。

## 当前边界与下一步

1. 增加 1K/4K prefill、64/128 output 的矩阵，确认长 prompt 下 TP prefill eager fallback 的吞吐与 TTFT。
2. 把固定 10 ms admission 演进成“目标 gear + deadline”的自适应窗口，争取收回 B4 TTFT，同时保持 token parity。
3. 对齐 usable KV tokens 后比较容量、抢占和 prefix-cache 压力；当前内存数不能直接作为容量排名。
4. 增加 24–72 小时 soak、worker kill、HCCL timeout、请求取消和 cache churn。
5. 精度门禁继续优先于性能发布：任何 kernel、collective 或 graph gear 变化都必须先过 logits/token parity。

## 原始证据

所有本轮 JSON 已纳入仓库，可直接审计每次 run 的 samples、p50/p95/p99、失败数和资源信息：

- [auto-infer 原始数据](profiling/qwen25-72b-tp8/final-2026-07-24/raw/auto-infer/)
- [vllm-ascend 原始数据](profiling/qwen25-72b-tp8/final-2026-07-24/raw/vllm-ascend/)
- [Omni-NPU 原始数据](profiling/qwen25-72b-tp8/final-2026-07-24/raw/omni-npu/)
- [auto-infer 正确性与 cache 验证](profiling/qwen25-72b-tp8/final-2026-07-24/raw/auto-infer/qwen25-72b-tp-prefill-graph-validation.json)

npu2 完整运行目录：`/data2/auto-infer-tp-graph-20260724/`。测试后 auto-infer Qwen2.5-72B TP8 服务保留在容器内 `0.0.0.0:18400`，便于继续发送真实请求。
