# 精度优先的生产级验收流程

本文定义 auto-infer 在完整生产拓扑上的发布验收流程。目标不是证明系统
“永不失败”，而是用可重复实验确认以下五个性质：结果正确、请求最终结束、
资源使用有界、故障可恢复、性能不会随运行时间持续恶化。

验收优先级固定如下，后一级不能覆盖前一级失败：

```text
P0 精度与结果完整性 > P1 稳定性与故障安全 > P2 性能与资源效率
```

只有 P0 全部通过，数据才有资格进入长稳和性能比较。更高吞吐、更低延迟或更低
显存不能抵消任何精度差异、silent corruption 或无法解释的结果不一致。

## 1. 验收原则

- 所有运行必须绑定代码 commit、容器镜像、CANN/torch-npu 版本、模型 revision
  和硬件拓扑；缺少其中任一项的数据不得用于发布结论。
- 每个请求必须具有唯一 ID，并进入完成、取消、显式失败或仍在运行四种状态之一。
- 性能、稳定性与精度使用同一套请求生成器和结构化结果格式，禁止人工摘录均值。
- eager 是 auto-infer 内部数值基线；外部框架用于交叉验证，不替代内部一致性门禁。
- 任何 silent corruption、永久挂起、资源泄漏或无法解释的 token 差异都阻断发布。
- 每个拓扑、模型、执行模式和关键 batch/长度边界必须先完成精度认证，再运行压测。
- 精度失败时立即停止该配置的稳定性和性能测试，保留全部 mismatch 证据并进入诊断。

请求账本必须始终满足：

```text
submitted = completed + cancelled + explicitly_failed + active
```

测试结束后 `active` 必须为零，KV block、请求状态、response queue 和临时缓冲区
必须回到稳态基线。

## 2. 拓扑分层

完整验收按相同 workload 逐层扩大，上一层未通过时不得进入下一层。

| 层级 | 拓扑 | 主要目标 |
|---|---|---|
| S0 | Host/MockExecutor | 生命周期、调度、不变量和错误传播 |
| S1 | 单卡 NPU | kernel、paged KV、graph、MTP 数值与资源稳定性 |
| S2 | 单机多卡 | TP/DP/EP、HCCL collective、进程生命周期与负载均衡 |
| S3 | 跨节点 | 节点间 collective、网络抖动、节点故障和恢复 |
| S4 | Prefill/Decode 分离 | KV 传输、block ownership、背压、取消和重试 |
| S5 | 完整生产副本 | 由部署层提供副本路由，并验证多租户混合流量、滚动升级和容量冗余 |

生产拓扑参数必须写入结果文件，包括节点数、每节点 NPU 数、TP/DP/EP/CP/SP、
Prefill 与 Decode 副本数、网络类型和每个进程的设备映射。

## 3. 测试矩阵

### 3.1 模型与执行路径

| 模型 | 必测路径 | 主要覆盖 |
|---|---|---|
| Qwen2/Qwen3 | paged、graph、sync、async | GQA、连续批处理、通用 decode |
| Moonlight | paged、graph | MLA、MoE、长 prefill、graph gear |
| MiMo | graph、graph_mtp | target/drafter、接受与拒绝、最终 token 边界 |

每个路径至少覆盖 greedy；支持随机采样的发布版本还必须覆盖 temperature、top-p、
top-k 和固定 seed。

### 3.2 请求形状

- 短输入/短输出、短输入/长输出、长输入/短输出、长输入/长输出；
- B1、B4、B16、B32、B64，以及请求持续到达的 continuous batching；
- prompt 长度位于 KV block size 的前后，例如 15、16、17；
- batch 位于 graph gear 的前后，例如 15、16、17 和 31、32、33；
- 最大上下文长度前一位、恰好等于最大长度，以及明确拒绝超长请求；
- KV 使用率 25%、50%、90% 和接近耗尽；
- prefix cache 全命中、部分命中、未命中、淘汰后复用；
- decode 扩展 block、强制 preemption、recompute 和请求恢复；
- MTP 接受 0/1 个 draft、最后只剩 1 个输出 token、block boundary 完成。

### 3.3 流量模型

| 流量 | 方法 | 验证目的 |
|---|---|---|
| 稳态 | 固定到达率，覆盖目标负载的 50%、80%、100% | 基准吞吐和延迟 |
| 阶梯 | 每 10–20 分钟提高到达率直至饱和 | 容量拐点和排队行为 |
| 突发 | 短时间注入 2–5 倍目标 QPS | admission、背压和恢复 |
| 潮汐 | 高低负载周期性交替 | 缓存、内存和 graph 状态回落 |
| 混合 | 随机 prompt/output、模型和租户 | 生产代表性与公平性 |
| 取消风暴 | 大量客户端在不同阶段断连 | 状态、KV 和线程释放 |

请求生成器必须记录计划到达时间和实际提交时间，区分服务延迟与压测端自身排队。

## 4. 精度先行的分层执行流程

### 4.1 P0 精度认证

每个拓扑层级进入负载测试前，按以下顺序认证：

1. 固定 tokenizer、模型 revision、dtype、权重 hash、prompt token IDs 和采样参数；
2. 用 auto-infer eager 单请求生成内部 oracle，保存逐 token IDs 和必要的 logits；
3. 依次运行 paged、graph、continuous batching、prefix cache、preemption、async
   和 MTP，同 oracle 逐请求、逐位置比较；
4. 覆盖 block、gear、batch、最大上下文和最终 token 等关键边界；
5. 运行固定 revision 的任务级精度集，并与已批准 baseline 比较；
6. 对外部框架差异保存首个 mismatch、top-k logits 和 top1/top2 margin；
7. 生成独立的 accuracy manifest。只有结论为 `PASS` 才能进入后续阶段。

P0 硬门槛：

| 项目 | 门槛 |
|---|---:|
| auto-infer eager 与 paged/graph greedy token 一致率 | 100% |
| B1 与 continuous batching 逐请求一致率 | 100% |
| prefix cache、preemption、sync/async 前后一致率 | 100% |
| MTP 与非 MTP target baseline 一致率 | 100% |
| MTP accepted draft 等于 target token | 100% |
| 错误、重复、丢失或乱序 token | 0 |
| 无法解释的外部 reference mismatch | 0 |
| 非量化任务精度相对已批准内部 baseline 的下降 | 0 |
| 量化任务精度下降 | 不超过测试前批准的模型级阈值 |

内部 token mismatch 不允许用“BF16 数值误差”直接豁免。只有跨框架差异可以在
完整 logits/margin 证据下归类为 near-tie；auto-infer 自身 eager 与生产路径仍须
满足 100% token 一致性。

### 4.2 PR 门禁

每个变更运行：

1. 全部 host 单元测试和静态检查；
2. 小模型 eager/graph 逐 token parity；
3. 10–20 分钟并发冒烟；
4. graph gear、KV block、取消和 preemption 的边界用例。

任何失败都阻断合并。

### 4.3 Daily

每晚运行 2–4 小时：

1. 单卡和单机多卡混合流量；
2. continuous batching、prefix cache、preemption 和取消并发发生；
3. Qwen、Moonlight、MiMo eager/graph/MTP 一致性；
4. 每分钟采集延迟分位数、吞吐、内存、显存、KV、队列和错误计数。

### 4.4 Release

每个候选版本在完整生产拓扑运行至少 24 小时或处理至少 100 万个请求，
以更严格的条件为准。顺序为：

1. Qwen 稳态与阶梯负载；
2. Moonlight MLA/MoE 长短请求混合；
3. MiMo MTP B4/B16 与动态请求；
4. 多模型、多长度、多租户综合流量；
5. 负载停止后继续观察 30 分钟，确认资源回到基线。

### 4.5 Weekly

每周运行 72 小时完整拓扑混合流量，并在稳态负载下周期性注入故障。故障注入
必须使用固定 seed 和时间表，确保版本之间可比较。

## 5. 稳定性硬门槛

本节属于 P1，只对已经通过 P0 精度认证的配置执行。

| 指标 | Release gate |
|---|---:|
| 错误、重复、丢失或乱序 token | 0 |
| 永久挂起请求 | 0 |
| 未预期进程退出 | 0 |
| 请求账本不平衡 | 0 |
| 合法准入负载下 NPU OOM | 0 |
| KV block、queue、thread 泄漏 | 0 |
| warm-up 后 24 小时显存净增长 | <5%，且不存在持续正斜率 |
| 首个稳态小时与最后小时的吞吐漂移 | <5% |
| 首个稳态小时与最后小时的 P99 漂移 | <10% |
| 稳态吞吐 CV | <1% |
| 单 worker 故障恢复 | <60 秒，或满足正式 SLA |

CV 只衡量相对波动，不能替代 P50、P95、P99、P99.9、最大延迟、超时率和
原始样本。出现离群值时必须保留该次结果并扩大样本复测，不得直接删除。

资源有界需要同时检查：

- NPU allocated/reserved memory 与分配失败次数；
- host RSS、pinned memory 和文件描述符；
- worker/thread 数和内部队列深度；
- KV free/used/cached block 与引用计数；
- graph capture 数、online capture 数和 fallback 计数；
- P/D 传输中的 block、请求和未完成事件数。

## 6. 精度一致性

本节定义 P0 的诊断和证据要求；第 4.1 节的硬门槛决定是否放行。

### 6.1 内部强一致性

greedy 模式要求逐请求、逐 token 100% 一致：

- graph 与相同 kernel/精度的 eager baseline；
- B1 与 continuous batching；
- prefix cache 开启与关闭；
- 无压力运行与发生 preemption/recompute 的运行；
- sync 与 async；
- MTP 与非 MTP target baseline。

MTP 还必须逐步验证：每个 accepted draft 等于 target token；拒绝后使用 target
correction token；请求完成不会污染同批其他请求；输出长度严格满足上限。

### 6.2 外部参考一致性

对 Transformers 和 vllm-ascend 保存：

- 完整 token IDs 与首个差异位置；
- top-1/top-5 token；
- logits 最大绝对误差、相对误差；
- `top1_logit - top2_logit` margin；
- 模型、dtype、batch shape、attention kernel 和采样配置。

BF16 下，如果 top-1/top-2 margin 极小，不同框架可能因合法数值路径选择不同
token。这类差异必须归类为 near-tie，并由 auto-infer eager/graph 内部一致性判断
是否为回归，不能只比较最终文本。

### 6.3 随机采样与任务精度

随机采样不能只要求不同实现逐 token 相等：固定相同 RNG 实现时检查 seed 重放；
跨实现使用 token 分布、KL/KS 检验和置信区间。发布还需运行固定 revision 的
GSM8K、MMLU、CEval、HumanEval、长上下文检索与结构化输出数据集。

- graph 与 eager 的任务得分必须一致或处于预先定义的统计误差内；
- MTP 与 target baseline 必须完全一致；
- 量化模型的允许下降必须在测试前定义，建议不超过 0.2–0.5 个百分点。

## 7. 故障注入

| 故障 | 注入阶段 | 必须观察的结果 |
|---|---|---|
| Prefill worker kill | 长 prompt 处理中 | 请求重试或明确失败，无 KV 泄漏 |
| Decode worker kill | 持续生成中 | 不产生静默错误 token，副本继续服务 |
| Router 重启 | 高并发 | 已接收请求有确定结果，新请求恢复准入 |
| NPU OOM/不可用 | prefill、decode、capture | 隔离故障设备，无永久挂起 |
| HCCL timeout | collective 执行中 | 超时可见，进程组安全重建或退出 |
| 节点掉线 | TP/EP 与 P/D 传输中 | 请求显式失败/重试，健康副本不中断 |
| 网络延迟、丢包、分区 | P/D KV 传输中 | 背压生效，不消费不完整 KV |
| 客户端断连 | prefill、decode、D2H copy | 请求取消，queue/KV/event 最终释放 |
| 取消风暴 | 各阶段随机发生 | 无死锁、负计数或其他请求污染 |
| 滚动升级 | 满载服务中 | 容量冗余有效，版本切换无 silent corruption |

故障恢复的通过条件不是“所有请求成功”，而是每个请求都有确定结局、错误可观测、
健康副本继续服务、资源最终释放，并在 SLA 内恢复正常吞吐。

## 8. 观测与结果格式

每次实验必须持久化原始 JSON，不只保存聚合表。至少包含：

```text
run_id, git_commit, image_digest, model_revision
torch_version, torch_npu_version, cann_version
topology, parallel_config, execution_mode, workload_seed
submitted, completed, cancelled, failed, active
input_tokens, output_tokens, accepted_drafts, verify_steps
request_latency_samples, ttft_samples, tpot_samples
throughput_samples, queue_depth_samples
host_rss_samples, npu_memory_samples, kv_block_samples
capture_counts, fallback_counts, preemption_counts
fault_timeline, recovery_times, token_digests, mismatch_records
```

报告必须同时给出原始样本、median、mean、stdev、CV、P50/P95/P99/P99.9、最大值
和首尾窗口漂移。所有失败请求必须保留 request ID、阶段、异常类型和关联日志。

## 9. 发布判定

发布需要同时满足：

1. 所有拓扑、模型和生产路径的 P0 accuracy manifest 均为 `PASS`；
2. 内部 greedy/MTP 强一致性和 MTP accepted-token 正确率均为 100%；
3. 外部差异均有 logits/margin 证据和明确分类；
4. PR、Daily、24 小时 Release 全部通过；
5. 没有 silent corruption、永久挂起和资源泄漏；
6. 所有稳定性硬门槛通过；
7. 故障注入达到正式恢复 SLA；
8. 原始结果、配置、日志和发布结论可由 commit 一键追溯。

P0 失败时版本直接判定为 `REJECTED`，不允许以性能提升、稳定性通过率或人工抽样
结果覆盖。P1 通过后才能发布性能结论；P2 性能不达标可以阻止发布，但不能修改
P0/P1 的验收标准。

72 小时 chaos soak 可以作为定期门禁，不必阻塞每个小版本；涉及 scheduler、KV、
graph ownership、distributed 或 P/D 的变更必须重新通过完整 chaos soak。
