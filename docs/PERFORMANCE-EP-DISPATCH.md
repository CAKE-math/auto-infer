# BF16 EP dispatch/combine 验证

日期：2026-07-21
目标：以 CANN fused dispatch/combine 替换 routed-output all-reduce，并与
omni-npu 的通信协议对齐。

## 已验证结果

| 项目 | 拓扑 | 结果 |
| --- | --- | --- |
| Host 全量回归 | CPU | 399 passed |
| 真实 HCCL dispatch/combine | 2 × Ascend 910B1，BF16 | 通过 |
| identity-expert 最大绝对误差 | EP2 | 0.0 |
| dispatch/combine 调用 | 每 rank | 1 / 1 |
| routed-output all-reduce | 每 rank | 0 |
| token identity | EP2 | 通过 |

Moonlight 单个 MoE 层在同一 NPU 1/2 卡对上的成对微基准如下。每组包含
10 次预热和 50 次交替采样；表中延迟取每次两个 rank 较慢值的中位数，因而
代表分布式 step 的关键路径。

| live tokens | all-to-all | 旧 all-reduce | 延迟下降 | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.134 ms | 2.704 ms | 21.1% | 1.267× |
| 8 | 2.069 ms | 2.551 ms | 18.9% | 1.233× |
| 16 | 2.094 ms | 2.630 ms | 20.4% | 1.256× |

三组数值对齐均通过；最大绝对误差分别为 `0.0009765625`、
`0.001953125`、`0.001953125`。这些数据证明真实 all-to-all 对 Moonlight
的 MoE 层有稳定收益，但不是完整请求吞吐或 TPOT 数据。

## 端到端诊断

EP2 eager、B1、单 token 的同卡对关键路径结果为：旧 all-reduce
`100.854 ms`，新 all-to-all `93.694 ms`，延迟下降 `7.1%`（取两个 rank
较慢值后求中位数）。32-token 请求的中位总耗时从 `2.773 s` 降至
`2.694 s`，方向性下降 `2.8%`。

但 32-token 数据不能作为正式吞吐验收：新旧路径在首 token 已分叉。首步
logits 显示旧路径的候选 `521/84` 为 `8.875/8.750`，新路径则均为
`8.875`；BF16 累积顺序把 `0.125` 的小 margin 变成平局。完整 prefill
对照的 logits cosine 为 `0.9998485`、最大绝对误差 `0.3672`，且两条路径
argmax 都为 `334`。证据指向数值边界翻转，而不是路由或通信错误；在完成
vLLM-Ascend token reference 前，不把端到端性能标记为最终通过。

EP2 结果文件位于 npu2：
`/data2/auto-infer-ep-dispatch-20260721/results/op-ep2-final-summary.json`。

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,4 MASTER_PORT=29721 HCCL_CONNECT_TIMEOUT=300 \
torchrun --standalone --nproc-per-node=2 scripts/verify_ep_dispatch.py \
  --model /data2/models/Moonlight-16B-A3B-Instruct \
  --ep-size 2 --operator-only --hidden-size 256 \
  --num-experts 256 --top-k 8 --output-dir results \
  --run-id op-ep2-final
```

## 架构结论

生产 MoE 路径现在是：gate → BF16 dispatch v2 → 本地 expert grouped GEMM →
combine v2 → shared expert。dispatch 返回的 expert counts 和通信 metadata
直接贯通计算与 combine；生产路径不再含 routed-output all-reduce。ACL Graph
和连续批处理使用固定地址的 bool active-token mask，padding token 不参与路由。

量化由 `MoeDispatchQuantization` 保留稳定接口；当前只接受
`quant_mode=0, scales=None`，其他策略在首次 forward 前显式拒绝。

## 尚未完成的设备验收

Moonlight EP2/EP4 的 eager/graph token identity 和端到端吞吐尚未形成有效
对照数据。EP1 reference 在单卡构建 64 个 BF16 expert 的 stacked weights 时
OOM，因此不能作为端到端基线；本文也不从单层结果外推完整请求收益。后续需要
以可运行的旧 EP2 all-reduce 版本和当前 EP2 all-to-all 版本，在相同卡对上分别
测量 TTFT、TPOT 和 tokens/s。vLLM-Ascend TP2 reference 已尝试启动，但测试
期间 NPU 1–4 被新任务占用、每卡仅余约 `8.88 GiB`，其结果尚未取得。
