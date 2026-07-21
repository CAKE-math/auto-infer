# auto-infer 系统机制补全 — 设计

Status: Draft · Date: 2026-07-16 · Target hardware: Ascend NPU (npu2)

## 背景与目标

对比 vLLM v1 后确认：auto-infer 的**功能面**砍对了（引导解码、LoRA、多模态、beam
等会往热路径塞 host 侧控制流和同步点，砍掉利于 NPU 性能主线），但几处**系统机制**
的省略不是免费的——它们恰是 vLLM *吞吐/鲁棒性*的来源，且其中数项在本仓库里"写了但
没接线"。本设计把这些机制真正实现好并在真机 NPU 验证。

本轮范围（5 项）：

1. 前缀缓存接线（当前 `match_prefix`/`register_prefix` 已写但 scheduler 从不调用）
2. 整批采样生效 + 完整版 SamplingParams（**仅采样数学全集**）
3. 调度策略改进（优先级 + TBT 平滑）
4. 抢占 / 重算（重算式 + 排空再抢占）
5. 真机 NPU（npu2）验证

**明确不做**（本轮排除，保持范围干净）：stop 字符串、logprobs/prompt_logprobs、
n>1/best_of、KV swap-to-host、引导解码、LoRA、多模态。

## 关键设计决策（已与用户确认）

- 抢占 KV 处理：**重算式**（不做 host swap）。
- 抢占 × 异步队列：**排空再抢占**（drain-then-preempt），异步快路径无压力时不受影响。
- SamplingParams：**采样数学全集**，不含需要独立子系统的基础设施档。
- 验证标准：**真机 NPU 运行**（目标 npu2），host 单测用 MockExecutor。

---

## 1. 前缀缓存接线

**现状问题**：`scheduler.py:95-97` 中 waiting 请求首次调度只做 `allocate(全长)`，从不
`match_prefix`；且 `kv_cache_manager.py:free()` 一到 refcount=0 立即把块退池并从
`_hash_to_block` 删除 → 跨请求几乎复用不到缓存。

**改动分三处：**

### 1a. KVCacheManager：可驱逐的空闲池（核心数据结构改动）
- `free()` 不再对 refcount=0 的块立即解注册 + 退回 `_free`；改为移入一个**保留 hash 的
  LRU 缓存池** `_cached`（`OrderedDict[block_id -> hash]`，含最近使用序）。
- `allocate`/`_alloc_one` 分配优先级：先取 `_free`（真正空闲）→ 池空时从 `_cached` LRU
  头驱逐（此时才 `del _hash_to_block[hash]`、`del _block_hash[block]`）→ 都空则触发抢占
  （见 §4）。
- `match_prefix` 命中 `_cached` 中的块时"复活"：从 `_cached` 取出、refcount 置 1、纳入活跃。
- 不变量：一个块要么在 `_free`、要么在 `_cached`（refcount=0 但可命中）、要么活跃
  （refcount≥1）；三态互斥。

### 1b. Scheduler：admit 时命中
- waiting 请求 `bt` 为空时：`matched = kv.match_prefix(r.prompt_token_ids)`；
  置 `r.num_computed_tokens = len(matched) * block_size`；剩余 `num_prompt_tokens -
  num_computed_tokens` 部分再 `allocate`，`bt = matched + new`。
- 与现有 chunked-prefill 天然组合：`remaining = num_prompt_tokens - num_computed_tokens`
  自动只覆盖未命中部分。
- 命中块只读（完整块注册后不可变，本引擎无对完整块的写），共享读安全，无需 COW。

### 1c. 注册完整块
- 块填满时 `register_prefix`：prefill 完成的完整块、以及 decode 过程中新填满的块。
- 实现上在 `append_slots` 产生新完整块后、以及 prefill 调度块分配后登记；`free` 时完整
  块进入 `_cached` 保留 hash（1a 已覆盖）。

**验收**：两个只差 max_tokens 的同 prompt 请求，第二个 `num_computed_tokens` 起点 =
prompt 全长（前缀全命中，prefill 近乎为 0）。

---

## 2. 整批采样 + 完整版 SamplingParams（采样数学全集）

**现状问题**：两个 executor 硬编码 `logits[...].argmax()`（`model_runner.py:91`、
`graph_decode_runner.py:251`），`collect()` 每请求一次 `.item()`（B 次 D2H 同步），
`SamplingParams.temperature` 无人读，写好的 `sampler.sample`（temp/top_k/top_p）未接线。

### 2a. SamplingParams 扩展（`engine/request.py`）
新增字段（全部为采样数学、逐 token 对 logits 处理）：
```
temperature, top_k, top_p, min_p,
presence_penalty, frequency_penalty, repetition_penalty,
logit_bias: dict[int,float] | None,
bad_words_token_ids: list[list[int]] | None,   # 预转好的 token id（不做字符串→token）
allowed_token_ids: list[int] | None,
min_tokens, ignore_eos
```
既有 `max_tokens/stop_token_ids` 保留。

### 2b. 向量化 logits 处理器（`layers/sampler.py`）
`sample(logits (B,vocab), params_batched)` 一次处理整批，全部 mask/加减，保持
graph-capturable、无 host 控制流：
- penalties：需要每请求"已出现 token 计数 / 集合"，由 executor 从 `Request.output_token_ids`
  （+ prompt 用于 presence）构造 `(B, vocab)` 计数/存在张量后传入。
- `logit_bias`/`bad_words`/`allowed_token_ids`：scatter 到 `(B,vocab)` mask。
- `min_tokens`/`ignore_eos`：当 `len(output) < min_tokens` 或 `ignore_eos` 时把
  eos/stop token logits 置 -inf。
- 然后 temperature → top_k → top_p → min_p → softmax → multinomial（temp≤0 走 greedy）。
- per-row 参数用张量（`temperature (B,)` 等），异构请求同批处理。

### 2c. 批采样接线（两个 executor + engine collect）
- executor：把各请求采样行一次 gather 成 `(B,vocab)`，构造 per-row 参数张量，调 2b 一次
  出 `(B,)` token 张量。
- `sampled_dev` 从"标量 dict"→"`(B,)` 张量 + rid 顺序表"。
- `sampled_of`（异步喂下一批）：`unbind`（view，**无同步**）成 per-rid device 标量。
- `collect`：一次 `.tolist()`（唯一 D2H 同步点），按顺序表映射回 rid。
- 采样仍在 ACL graph 外（现状即如此），不影响捕获。

**验收**：批采样结果与逐请求采样一致（固定 seed）；`collect` 每步只 1 次 D2H。

---

## 3. 调度策略改进（`engine/scheduler.py` + `config`）

**现状**：FCFS + 固定 decode-先-prefill + 单一 `max_num_batched_tokens` 预算。

- **优先级**：`Request` 加 `priority: int = 0`（默认 FCFS）；waiting 出队按
  `(−priority, 到达序)`。
- **TBT 平滑**：`SchedulerConfig` 加 `long_prefill_token_threshold`（每步 prefill token
  单独封顶）；decode 已先调度受保护，仅给 prefill 份额封顶，避免大 prefill 抬高单步延迟。
- 保持简单：不做 swap、不做复杂公平/加权队列。

**验收**：高优先级请求先于同时到达的低优先级出队；单步 prefill token 不超过阈值。

---

## 4. 抢占 / 重算（重算式 + 排空再抢占）

**现状**：`kv_cache_manager` 分配不足直接 `raise MemoryError`。

### 4a. Request 重算语义（`engine/request.py`）
- 加 `num_prefill_tokens: int`（默认 = `num_prompt_tokens`）。prefill 循环改用它。
- 抢占时：`num_prefill_tokens = num_tokens`（prompt + 已生成），`num_computed_tokens = 0`，
  释放全部 KV（进 `_cached`），退回 waiting **队首**。已生成 `output_token_ids` 保留。
- 恢复：重新 prefill 整段（可命中前缀缓存复用 prompt 前缀）后续 decode。

### 4b. Scheduler 抢占
- victim 选择：**LIFO**，抢最近 admit 的 running 请求（对齐 vLLM v1）。
- `preempt_one()`：选 victim、走 4a 释放退回；`schedule()` 在 decode 需要新块但分配不到时
  置 `SchedulerOutput.needs_preemption = True`（**只在压力下置位**）。

### 4c. Engine 排空再抢占（`engine/engine_core.py`）
- `_step_async`：`schedule()` 返回 `needs_preemption` 时——停止 admit 新批 → drain 在途
  队列（`_queue` 深度 ≤ `async_batches`，逐个 `collect` + `_finalize`）→ 干净状态下
  `scheduler.preempt_one()` → 重新 `schedule()`。
- `_step_sync`：无在途队列，直接 `preempt_one()` 后重排。
- 异步无压力快路径完全不变（`needs_preemption` 仅压力下为真）。
- 不变量：抢占只在 `_queue` 为空（无在途 token）时执行 → 不会释放在途批引用的块。

**验收**：小 `num_blocks` + 高并发触发抢占；被抢占请求恢复后输出与不抢占时逐 token 一致。

---

## 5. 验证

### 5a. Host 单测（MockExecutor，无需 NPU）
- 前缀命中跳过重算；LRU 驱逐正确（池满时驱逐最久未用、hash 解注册）。
- 批采样 rid 映射与逐请求一致；`collect` 单次 D2H。
- 优先级出队顺序；prefill 份额封顶。
- 抢占 victim 选择 + 重算状态机；drain-then-preempt 不损坏在途批（异步下抢占前队列已空）。

### 5b. 真机 NPU（npu2）
- 现有 smoke 全过（回归）。
- 新增：共享 system prompt 脚本展示前缀缓存命中（第二请求 prefill token 数骤降）。
- 新增：小 `num_blocks` + 高并发脚本触发真实抢占，验证输出正确。
- npu2 访问方式在实现阶段先确认（README 记录 npu2 存在、RoCE NIC down；单节点足够）。

## 影响文件

- `engine/kv_cache_manager.py`（可驱逐池 — §1a）
- `engine/scheduler.py`（前缀命中 §1b、优先级/TBT §3、抢占 §4b）
- `engine/request.py`（SamplingParams §2a、num_prefill_tokens/priority §4a/§3）
- `engine/engine_core.py`（排空再抢占 §4c）
- `layers/sampler.py`（向量化 logits 处理器 §2b）
- `worker/model_runner.py` + `worker/graph_decode_runner.py`（批采样接线 §2c）
- `config/__init__.py`（`long_prefill_token_threshold` §3）
- `tests/`（§5a）、`scripts/`（§5b 两个新脚本）

## 实现顺序（按依赖与风险）

1. 前缀缓存（§1，独立、低风险，为 §4 重算复用打基础）
2. 批采样 + SamplingParams（§2，低风险）
3. 调度策略（§3，低风险）
4. 抢占 / 重算（§4，最高价值、最难，叠加在 §1 之上）
5. 真机验证（§5）
