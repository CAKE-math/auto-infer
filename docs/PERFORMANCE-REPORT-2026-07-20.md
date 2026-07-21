# auto-infer 简明性能报告

测试日期：2026-07-20
测试设备：单张 Ascend 910B1（npu2）
精度与解码：BF16、greedy、ignore EOS

## 指标与批次定义

- **B（Batch Size）**：一次推理调用中同时处理的独立请求数量。
- **B4**：一个批次包含 4 个请求；**B16**：一个批次包含 16 个请求。
- **tok/s**：批次内所有请求生成的 token 总数除以该批次的端到端耗时，
  是整卡聚合吞吐，不是单请求吞吐。例如 B16 每个请求生成 32 tokens 时，
  吞吐计算的分子为 `16 × 32 = 512 tokens`。
- **TTFT**：单请求（B1）从提交到生成首个 token 的时间；越低越好。
- **TPOT**：首个 token 之后，每生成一个 token 的平均时间；越低越好。
- **CV（变异系数）**：用于衡量多次测试结果的相对波动，计算公式为
  `CV = 标准差 ÷ 平均值 × 100%`。例如平均吞吐为 1,000 tok/s、标准差为
  5 tok/s，则 CV 为 0.5%，表示运行间波动约为平均值的 0.5%。CV 越低，
  性能越稳定；本报告以 CV < 1% 作为稳定性参考门槛。CV 不能反映少量严重
  卡顿，因此长时间服务测试仍需同时观察 P95/P99 尾延迟和原始样本。

## 结论

在已完成同配置验证的 Qwen3、Moonlight 和 MiMo MTP 工作负载中，
auto-infer 的吞吐、TTFT/TPOT 和稳定性均领先有效对照结果。输出正确性方面，
auto-infer 与 vllm-ascend 的 Qwen3、Moonlight 和 MiMo B16 digest 一致。

## 原始结果

### Qwen3-0.6B：通用推理

配置：一次同时处理 16 个请求（B16），每个请求输出 128 tokens，可用 KV
容量均为 14,464 tokens。TTFT 和 TPOT 使用单请求（B1）测量。

| 框架 | B1 TTFT | B1 TPOT | B16 聚合吞吐 | 加载时间 | 峰值显存 | 吞吐 CV |
|---|---:|---:|---:|---:|---:|---:|
| **auto-infer** | **5.90 ms** | **5.53 ms** | **2,259.2 tok/s** | **1.48 s** | **2.787 GiB** | **0.700%** |
| omni-npu 0.14.0 | 52.84 ms | 6.52 ms | 1,966.9 tok/s | 52.04 s | 9.731 GiB | 0.745% |
| vllm-ascend 0.20.2 | 18.99 ms | 17.74 ms | 847.9 tok/s | 44.48 s | 2.797 GiB | 0.947% |

相对吞吐提升：对 omni-npu 为 **+14.9%**，对 vllm-ascend 为 **+166.5%**。
auto-infer 与 vllm-ascend digest 一致；omni-npu 输出长度一致但 digest 不同。

### Moonlight-16B-A3B：MLA + MoE

配置：一次同时处理 4 个请求（B4），每个请求输出 32 tokens；一次 warm-up、
五次测量。Cold/Warm TTFT 和 TPOT 使用单请求（B1）测量。

| 框架 | B1 Cold TTFT | B1 Warm TTFT | B1 TPOT | B4 聚合吞吐 | CV |
|---|---:|---:|---:|---:|---:|
| **auto-infer** | **32.00 ms** | **25.05 ms** | **13.63 ms** | **228.99 tok/s** | **0.715%** |
| vllm-ascend | 22,717.15 ms | 47.75 ms | 42.05 ms | 91.60 tok/s | 0.850% |
| omni-npu | — | — | — | — | graph capture 失败 |

auto-infer 吞吐为 vllm-ascend 的 **2.50 倍**，Warm TTFT 降低 **47.5%**，
TPOT 降低 **67.6%**；两者 digest 均为 `599c9e73d403b339`。

### MiMo-7B：MTP 推测解码

配置：每个请求输出 32 tokens。B4 一次处理 4 个请求、共生成 128 tokens；
B16 一次处理 16 个请求、共生成 512 tokens。auto-infer MTP 接受率 79.41%，
平均每个验证 step 输出 1.794 tokens。

| 框架 | B4 聚合吞吐 | B16 聚合吞吐 | 稳定性 | 正确性 |
|---|---:|---:|---:|---|
| **auto-infer** | **250.55 tok/s** | **895.51 tok/s** | B4 CV 0.509%；B16 CV 0.240% | B4/B16 均通过 |
| vllm-ascend | 176.8 tok/s | 673.6 tok/s | B4 CV 0.69%；B16 CV 0.96% | B16 digest 一致 |
| omni-npu | 101.1 tok/s | — | B4 CV 1.33% | 仅 eager fallback 可运行 |

auto-infer 对 vllm-ascend 的吞吐领先为：B4 **+41.7%**、B16 **+32.9%**；
对 omni-npu B4 为 **2.48 倍**。B16 digest 为 `f660d7348f95bc30`。

## 关键发现

1. auto-infer 在三个已验证工作负载上均取得最高有效吞吐，同时保持低于 1% 的最终稳定性 CV。
2. Moonlight 的优势同时覆盖冷启动、稳态首 token 和逐 token 延迟，不只是批量吞吐。
3. MiMo 的提升来自 two-stage graph MTP、连续批处理和持久化 metadata/staging；生产路径没有 eager/fused fallback。
4. Qwen3 对 omni-npu 的 digest 不一致，因此该组只支持性能比较，不支持跨框架 token 等价声明。

## 后续验证建议

- 修复 omni-npu Moonlight rotary graph-capture 问题后补齐三方 Moonlight 对比。
- 增加 B32/B64、长上下文和持续一小时的尾延迟稳定性测试。
- 获得多层 MTP checkpoint 后验证并实现多层 recurrence；当前实现会对不支持的几何形状显式报错。

原始数据与日志保存在 npu2：
`/data2/auto-infer-decode-performance/logs/final-architecture-20260720/` 和
`/data2/auto-infer-decode-performance/results/final-20260720/`。
