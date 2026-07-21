# Native Async Serving 验收报告

日期：2026-07-20
设备：npu2，单张 Ascend 910B1（0 号卡）
模型：Moonlight-16B-A3B-Instruct，BF16 greedy，32 个输出 token
版本：`55973c8`

## 结论

原生异步文本 Serving 的架构、正确性和短时稳定性门禁通过。离线与在线
32 个 token 完全一致，流式与非流式文本完全一致，B1/B4/B16 均无错误；
最终 60 秒并发 16 soak 完成 1,122 个请求，失败和遗留请求均为 0。

24 小时 soak 尚未完成，因此本文不把当前结果表述为完整生产稳定性验收，
也不提前给出与 vllm-ascend、omni-npu 的在线 Serving 排名。

## 测试口径

- B1、B4、B16 分别表示客户端同时保持 1、4、16 个生成请求。
- TTFT 是从客户端发出请求到收到首个非空流式文本片段的时间。
- 吞吐是所有成功请求生成 token 的聚合速率，不是单请求速率。
- 变异系数（CV）是标准差除以均值，用于表示相对波动；越低越稳定。
- KV 配置为 256 个可用块、16 tokens/块，即 4,096 个可用 KV tokens；
  graph runner 另有 32 个 scratch 块。`max_model_len=512`。

## 正确性

| 门禁 | 结果 |
|---|---:|
| `/health`、`/v1/models` | PASS |
| `/v1/completions` 非流式 | PASS |
| `/v1/completions` SSE | PASS |
| `/v1/chat/completions` | PASS |
| 非法参数返回 400 | PASS |
| 流式与非流式文本一致 | PASS |
| B1/B4/B16 greedy 输出一致 | PASS |
| 在线与离线 token IDs 一致 | PASS，32/32 |
| 在线与离线文本一致 | PASS |

在线与离线 token digest 均为
`599c9e73d403b339f6e89a93678cf265ac65e3a6967c6c1175d98c83eb6bb844`。

## 在线性能

每组先 warm-up 1 次。B1 测 8 个请求，B4 测 16 个请求，B16 测 32 个请求。
性能测试关闭 SSE coalescing，保证每个流式 chunk 最多对应一个 token；脚本若
检测到多 token chunk 会直接判失败。服务端 CPU/RSS 通过必填的 server PID 采集。

| 并发 | 完成/失败 | p50 TTFT | p99 TTFT | p50 ITL | p50 E2E | 聚合吞吐 | 服务端 CPU | 服务端峰值 RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 8 / 0 | 31.01 ms | 31.60 ms | 13.96 ms | 464.77 ms | 68.80 tok/s | 6.00 s | 2.05 GiB |
| B4 | 16 / 0 | 99.87 ms | 107.14 ms | 16.31 ms | 607.53 ms | 209.40 tok/s | 3.81 s | 2.06 GiB |
| B16 | 32 / 0 | 133.89 ms | 208.83 ms | 21.74 ms | 841.06 ms | 599.45 tok/s | 2.62 s | 2.06 GiB |

这些数值包含 HTTP、异步 tokenizer、调度、推理、增量 detokenize 和 SSE，
不能与已有的纯 Engine benchmark 直接混用。

## 稳定性与可观测性

- 最终 60 秒、并发 16：提交 1,122，完成 1,122，失败 0，active 0。
- 请求延迟 CV：1.06%，低于 3% 的短时稳定性门槛。
- `/metrics` 返回 200。
- `http_parse`、`admission_wait`、`tokenize`、`engine_queue`、`prefill`、
  `decode`、`sse_send`、`ttft`、`itl`、`e2e` 均使用主机时间戳采集；
  不触发 NPU 同步。真实 Moonlight 请求已确认前四个此前缺失的阶段均产生样本。
- 最新源码在 npu2 容器内全量回归：340 passed，14 个既有弃用警告。
- 四轮独立代码审查最终结论：无 Critical/Important；特别覆盖 async
  scheduling placeholder、queued abort、阻塞关闭和指标口径。

## 证据位置

远端根目录：`/data2/auto-infer-native-serving-20260720`

- `results/moonlight-serving-post-review-correctness.json`
- `results/moonlight-online-offline-greedy.json`
- `results/moonlight-serving-post-review-b1.json`
- `results/moonlight-serving-post-review-b4.json`
- `results/moonlight-serving-post-review-b16.json`
- `results/moonlight-serving-post-review-soak-60s.json`
- `results/moonlight-chat-smoke.json`
- `logs/post-review-host-tests.log`

## 后续门禁

完成 24 小时 soak 后，再在同卡、同模型、同请求集、同 KV 容量和同并发口径下，
依次运行 auto-infer、vllm-ascend、omni-npu。只有三方原始样本齐全后才给出最终排名；
同时检查 RSS/NPU 内存斜率、线程数、KV 回收、错误率和慢客户端隔离。
