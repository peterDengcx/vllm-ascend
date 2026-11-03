# MTP → verify 空泡实验修改说明

代码位置：`/home/d/workspace/vllm-ascend`
分支：`codex/mtp-verify-bubble`；基线提交：`56af62b51`。适用范围：Model Runner V1。V2 不受这些改动影响。当前 profile 的 runner 和模型与此工作树对应关系仍需在服务运行时确认。

## 改动及触发条件

1. `vllm_ascend/worker/v1/model_runner.py`：为 accepted-token event 和 valid-count event 的 Host 同步添加 `mtp_gap.accepted_tokens_event_wait`、`mtp_gap.valid_count_event_wait` profiler 区间。同步本身保留。它们只记录 Host 在调用中的时间；如果不进入相应分支，就没有对应区间。
2. `VLLM_ASCEND_EXPERIMENT_MTP_EARLY_DRAFT=1`：在 V1、`method=mtp`、异步调度、PP1、padded drafter 且无结构化输出或要求 CPU output token IDs 的条件下，在 bookkeeping 前发起 MTP；其他条件沿用原顺序。目标是让 NPU draft 与 Host bookkeeping 有机会重叠。原默认值为 `0`。注意原有代码尚有 bookkeeping 与 proposer 重构 TODO，字段共享、批次变更、状态一致性没有 NPU 验证；这只是实验路径，不应直接用于生产。
3. `VLLM_ASCEND_EXPERIMENT_MTP_EMPTY_BUFFER=1`：V1 MTP 的多步 draft token 临时张量使用 `torch.empty` 代替 `torch.zeros`。每行在本路径中按步骤写入后才被读取；目标是减少初始化 Fill，默认值 `0`。不同图捕获、padding 和异常路径仍须验证；这可能缩短 draft 内部时间，并不保证缩短交界空泡。
4. 两个开关注册在 `vllm_ascend/envs.py`，在启动服务的进程环境中设置；重启服务生效。有效值 `0` 或 `1`，均不是敏感信息。默认关闭，不需要改原服务命令。

## 运行前确认

在 WSL：

```bash
cd /home/d/workspace/vllm-ascend
git branch --show-current
git rev-parse HEAD
git diff --stat
```

确认服务实际导入这个工作树，而不是容器里另一个已安装版本；在**服务实际使用的 Python 环境**执行：

```bash
python -c 'import vllm_ascend, vllm_ascend.worker.v1.model_runner as m; print(vllm_ascend.__file__); print(m.__file__)'
```

需核实实际运行的是 V1、`method=mtp`、PP1、异步调度、padded drafter；如未命中条件，提前 draft 开关不触发。未安装或无法导入时先配置该环境，不要将本机导入成功误当作服务容器导入成功。

## 单变量 A/B

保持原有模型、TP2、权重、输入输出长度、MTP 深度、图模式、并发、prefix cache 及服务参数不变。使用用户确定的 case；每个 case 请求数始终为并发数的 4 倍。每组先关闭 profiler 测至少三次并记录中位数，再单独采短 trace。不要根据本文自行替换服务命令或修改其他参数。

| 实验 | 服务环境变量 | 作用 |
|---|---|---|
| B0 | 两个开关均设为 `0` | 基线 |
| B1 | `VLLM_ASCEND_EXPERIMENT_MTP_EARLY_DRAFT=1`，另一项为 `0` | 测提前 draft |
| B2 | `VLLM_ASCEND_EXPERIMENT_MTP_EMPTY_BUFFER=1`，另一项为 `0` | 测减少 draft buffer Fill |
| B3 | 两项均为 `1` | 仅 B1、B2 分别正确后测组合 |

同一 shell 中示例：`export VLLM_ASCEND_EXPERIMENT_MTP_EARLY_DRAFT=1`；另一个开关显式设 `0`。开关必须传入实际服务容器/进程；修改后重启服务。每次记录启动命令、配置、版本、负载、输出 token、接受长度、成功数、TPOT、output token/s、显存峰值及错误日志。

## 正确性与性能门槛

- 先做小负载功能对照：固定输入和采样设置，比较生成结果、接受/拒绝统计、请求结束、多请求批次变化、最长上下文附近的行为；随机采样还需要质量/分布验证。TP2 两 rank 均检查异常和死锁。不要把文本完全一致作为随机采样的唯一正确性标准。
- trace 对齐同一请求/轮次的 MTP 最后设备任务与下一次 verify 首个设备任务；结合现有 `draft_token`、`prepare input`、`forward` 和新增两个 event wait 区间。Host 区间不是设备完成时间；需要 NPU task/event 对齐。查看双卡 Free、rank skew、API 提交、MTP 每步时长及 Fill 次数。
- B1 成功要求 bookkeeping 与 MTP 有可核实重叠，且端到端 TPOT/吞吐改善；交界空泡和 MTP 启动前耗时要分别计算。
- B2 成功要求相应 Fill/初始化开销减少、输出与状态正确且端到端改善；不得仅凭少一个 Fill 就宣布成功。
- 修改前后 profiler-off 同条件各至少三次，中位数与波动一起报告。profile 缺 Step Id 时不能伪造 step；需要真实轮次标记。

## 尚未改动的候选

- accepted token event 同步：保证上轮状态就绪，先用新增标记测量；不能直接删除。是否可移动独立准备工作，需先证明字段依赖。
- `block_table_tensor.clone()`：保护不同 draft step 的 metadata；还缺引用生命周期与图回放证据，暂未改动。
- ACLGraph API 内部等待、TP collective rank skew、V2 上游 scheduler/runner：缺少目标运行路径和因果证据，尚未修改。
- KV 全量物化：属于另一条 attention 数据搬运路径，且与严格 MTP→verify 空泡的时间归属未闭合，未在本分支修改。

## 当前验证状态

已做 Python 语法编译和 `git diff --check`；没有在 NPU 环境运行正确性、profile、性能对照，也没有验证服务实际加载该分支。所有性能收益均待实测。
