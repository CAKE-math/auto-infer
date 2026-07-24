# Qwen2.5-72B TP8 三框架在线服务对比

日期：2026-07-24  
设备：单机 8 × Ascend 910B1  
模型：`/data1/models/Qwen2.5-72B-Instruct`  
精度：BF16  
服务：OpenAI `/v1/completions`，streaming，greedy，`ignore_eos=true`

## 结论

这次大模型 TP8 对比的排名与此前 Qwen3-0.6B 单卡结果不同：

1. **Omni-NPU 在 B1、B4、B16 吞吐和 ITL 上均为第一。**
2. **vllm-ascend 吞吐第二，auto-infer 第三。**
3. auto-infer 的 B4/B16 TTFT p50 最低，但逐 token 解码显著更慢，无法转化为端到端吞吐优势。
4. 当前首要性能缺口不是 serving API，而是 **TP decode 仍运行 paged eager**。auto-infer 会显式拒绝尚未通过数值门禁的 TP graph；另外两框架在本次运行中使用了 ACL Graph。

因此，不能把 auto-infer 在 Qwen3 单卡、Moonlight 和 MiMo MTP 上的领先外推到 Qwen2.5-72B TP8。当前生产 TP8 路径已正确运行，但性能尚未达到目标。

## 测试口径

- 相同 checkpoint、机器、8 张 NPU、BF16、`max_model_len=4096`、`max_num_seqs=32`。
- 实际 prompt 长度经 Qwen2.5 tokenizer 复核为 **126 tokens**；原始 benchmark JSON 的描述字段写为 128，属于 metadata 舍入/标注误差，不影响实际发给三框架的相同文本。
- 每个请求固定输出 64 tokens。
- B1：并发 1，共 8 个测量请求。
- B4：并发 4，共 20 个测量请求。
- B16：并发 16，共 32 个测量请求。
- 每组先做 2 个 warm-up 请求；吞吐为全部完成 token 数除以该组 wall time。
- TTFT、ITL 和 E2E 来自每个 streaming 请求的原始时间样本。
- 三框架均为 0 failed、0 rejected。

版本：

- auto-infer：`64a9b40`
- vllm-ascend：vLLM 0.20.2 + vllm-ascend 0.20.2rc1
- Omni-NPU：vLLM 0.14.0 + omni-npu 0.2.0，按其运行契约启用 `OMNI_NPU_VLLM_PATCHES=ALL`

## 原始性能表

### 聚合输出吞吐

单位：tok/s，越高越好。

| 并发 | auto-infer | vllm-ascend | Omni-NPU | 第一名 |
|---|---:|---:|---:|---|
| B1 | 5.36 | 18.23 | **31.68** | Omni-NPU |
| B4 | 20.85 | 63.98 | **111.21** | Omni-NPU |
| B16 | 77.89 | 115.27 | **352.80** | Omni-NPU |

相对 auto-infer：

| 框架 | B1 | B4 | B16 |
|---|---:|---:|---:|
| vllm-ascend | 3.40× | 3.07× | 1.48× |
| Omni-NPU | 5.91× | 5.33× | 4.53× |

auto-infer 要追平当前 Omni-NPU 的 B16 结果，需要把 77.89 tok/s 提升到 352.80 tok/s，即约 **4.53×**。

### TTFT p50

单位：ms，越低越好。

| 并发 | auto-infer | vllm-ascend | Omni-NPU |
|---|---:|---:|---:|
| B1 | 203.28 | 204.68 | **165.93** |
| B4 | **201.94** | 519.32 | 311.40 |
| B16 | **249.83** | 584.08 | 436.26 |

auto-infer 的请求进入执行、prefill 和首 token 路径并不慢；B16 TTFT p50 比 Omni-NPU 低 42.7%，比 vllm-ascend 低 57.2%。

### ITL p50

单位：ms，越低越好。

| 并发 | auto-infer | vllm-ascend | Omni-NPU |
|---|---:|---:|---:|
| B1 | 190.13 | 50.83 | **29.35** |
| B4 | 186.90 | 54.03 | **30.74** |
| B16 | 198.06 | 54.62 | **38.08** |

B16 下 auto-infer 的 ITL 是 vllm-ascend 的 3.63×、Omni-NPU 的 5.20×。这直接解释了吞吐排名：差距集中在稳态 decode，而不是服务接入层。

### E2E p50

单位：s，越低越好。

| 并发 | auto-infer | vllm-ascend | Omni-NPU |
|---|---:|---:|---:|
| B1 | 12.03 | 3.49 | **2.02** |
| B4 | 12.08 | 3.98 | **2.27** |
| B16 | 12.83 | 8.82 | **2.86** |

## 为什么 auto-infer 当前落后

### 已由代码和运行日志确认

1. `TpServingConfig` 对 `graph` 和 `graph_mtp` TP 模式 fail-fast，当前部署使用 `--mode paged`。也就是说，80 层 Qwen2.5-72B 的每个 decode token 都经过 eager TP 路径。
2. vllm-ascend 日志明确显示 PIECEWISE ACL Graph、`enable_npugraph_ex`、图编译和七个 graph batch size。
3. Omni-NPU 使用自己的 NPU attention backend、补丁和 graph/fusion 路径。
4. auto-infer 的单卡 graph pipeline、zero-host-bubble async、持久 staging 和 captured epilogue 尚未贯通到 TP executor。
5. auto-infer 已实现 QKV/gate-up 的 TP shard 后打包和 TP all-reduce，但打包权重只能减少算子数量，不能消除 eager launch、80 层逐步 dispatch 和 collective 排序开销。

### 从数据推断

- auto-infer TTFT 不差但 ITL 很差，说明 scheduler、HTTP/SSE 和 admission 不是主瓶颈。
- B1 ITL 已达 190 ms，负载尚未形成大规模排队，因此 continuous batching 也不是根因。
- 差距最符合“TP eager device critical path 过长”：kernel launch、HCCL collective、同步边界和未捕获 epilogue 在每个 token 上重复。

这些因果解释与代码和日志一致，但还需要 TP8 Chrome/Perfetto trace 和逐层 ablation 才能把差距精确拆成 ACL Graph、HCCL、attention、MLP、lm_head 各自占比。

## 输出正确性

相同 21-token parity prompt、64-token greedy 输出：

| 对比 | 公共 token 前缀 | 64-token 完全一致 |
|---|---:|---|
| auto-infer vs vllm-ascend | 62 | 否 |
| auto-infer vs Omni-NPU | 56 | 否 |
| vllm-ascend vs Omni-NPU | 56 | 否 |

三者均返回 64 tokens，输出语义连贯。分歧发生在长公共前缀之后，符合 BF16、不同 attention/graph/collective 数值路径在近似并列 logits 上发生 greedy 分叉的现象；但在 reference logits 和长数据集精度门完成前，本报告只声明服务正确完成，不声明 bitwise 精度等价。

## 内存观察

测量期间每 rank 的 `npu-smi` 进程内存大致为：

- auto-infer：25.4–26.5 GiB
- vllm-ascend：约 49.6 GiB
- Omni-NPU：约 52.5 GiB

这组内存值**不能作为严格容量排名**：auto-infer 使用固定 `num_blocks=2048`，vllm-ascend/Omni-NPU 使用 `gpu_memory_utilization=0.80` 并预留了更大的 KV cache。它只能说明当前部署配置下 auto-infer 明显更省 HBM；后续必须对齐 usable KV tokens 后再做正式内存比较。

## 下一步性能收敛顺序

1. 为 BF16 dense Qwen TP2–TP8 建立数值门禁，并让 TP paged executor 进入 ACL Graph。
2. 采集三个框架相同 B1/B16 的 TP8 trace，拆解每 token 的 model compute、HCCL、host gap 和 sampler/epilogue。
3. 将单卡的持久 staging、graph 内 lm_head/argmax、event 排序和双缓冲 metadata 复用到每个 TP rank。
4. 对 TP row-parallel all-reduce 做独立基准，核对 collective 是否与后续计算重叠，以及是否存在每层不必要同步。
5. 对齐 usable KV capacity 后重跑 B1/B4/B16，并增加 1024/128、4096/128 长上下文场景。
6. 通过 reference logits、固定 token corpus 和任务精度集后，才发布最终三框架排名。

## 原始证据

本地可审计 JSON：

- [`raw/auto-infer-b1.json`](profiling/qwen25-72b-tp8/raw/auto-infer-b1.json)
- [`raw/auto-infer-b4.json`](profiling/qwen25-72b-tp8/raw/auto-infer-b4.json)
- [`raw/auto-infer-b16.json`](profiling/qwen25-72b-tp8/raw/auto-infer-b16.json)
- [`raw/vllm-ascend-b1.json`](profiling/qwen25-72b-tp8/raw/vllm-ascend-b1.json)
- [`raw/vllm-ascend-b4.json`](profiling/qwen25-72b-tp8/raw/vllm-ascend-b4.json)
- [`raw/vllm-ascend-b16.json`](profiling/qwen25-72b-tp8/raw/vllm-ascend-b16.json)
- [`raw/omni-npu-b1.json`](profiling/qwen25-72b-tp8/raw/omni-npu-b1.json)
- [`raw/omni-npu-b4.json`](profiling/qwen25-72b-tp8/raw/omni-npu-b4.json)
- [`raw/omni-npu-b16.json`](profiling/qwen25-72b-tp8/raw/omni-npu-b16.json)
- 三份 `*-parity-response.json`

npu2 原始目录：

- `/data2/auto-infer-tp8-threeway-20260724/results/`
- `/data2/auto-infer-tp8-threeway-20260724/logs/`

测试结束后，auto-infer Qwen2.5-72B TP8 服务已恢复到容器内 `0.0.0.0:18400`，`/health` 返回 200。
