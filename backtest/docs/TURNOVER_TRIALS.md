# 十分组换手实验：规则、复现与服务器位置

[返回回测总览](../README.md) · [全配置结果](../results/20260917_turnover_trials/README.md) · [固定协议](../src/protocol_turnover_trials.json)。

本批已在 AutoDL 完成 **21 个持仓配置 × 4 种成本 = 84 组**。完整结果位于 `backtest/results/20260917_turnover_trials/`。模型、分数、行情、股票池和 2025-01-02 至 2026-08-28 区间固定，比较持仓规则；没有重新训练。

## 策略细则

### 每日 Q10−Q1：1 个配置

`daily_decile` 每天按上一交易日收盘后的原始分数排序，买最高 10%、卖空最低 10%，每侧目标 100 只、等权。下一日开盘调仓，多头目标为扣费后净值 +100%，空头为 −100%。并列分数按稳定的股票代码顺序处理。

### 排名缓冲：4 个配置

`buffer_10_15`、`buffer_10_20`、`buffer_10_25`、`buffer_10_30` 的每侧目标数量都是 100 只，仅改变旧持仓可保留的排名范围。

以 30% 保留带为例：旧多头在最新分数最高 300 只内、旧空头在最低 300 只内时优先保留；不足 100 只再从各自高/低分方向补齐，每侧仍等权调整。退出保留带的股票换出。保留带内优先保留并不等于权重不动，价格变化后恢复等权也可能产生交易。

### 低频调仓：10 个配置

`every_2d_phase0/1` 共 2 个，`every_3d_phase0/1/2` 共 3 个，`every_5d_phase0/1/2/3/4` 共 5 个。调仓日使用最新可用分数重选 Q10/Q1，两次调仓之间保持股数。

令 `t=0` 为初始现金日 2025-01-02，`k` 为调仓间隔，`phase` 为相位：

- `t=1`：所有相位都先全量建仓。
- `t>1` 且非期末：满足 `(t - 1 - phase) % k == 0` 才调仓。
- 最后一个交易日：所有相位都尝试退出，不受调仓间隔限制。

例如每 3 日调仓，建仓后的调仓下标分别为：相位 0 的 `t=4,7,10,...`，相位 1 的 `t=2,5,8,...`，相位 2 的 `t=3,6,9,...`。相位改变的是建仓后的日历；所有相位都报告。非调仓日不恢复多空各 100%，敞口会随价格漂移。

### 渐进调仓：3 个配置

`partial_25`、`partial_50`、`partial_75` 分别取调整系数 α = 25%、50%、75%。每日先将旧持仓按多空方向各自归一，再与最新 Q10/Q1 目标做 `(1−α) × 旧权重 + α × 新权重` 的混合；同一股票相反方向互相抵消后，再将正、负权重分别归一到 +100%、−100%。

首次建仓全量进入，前一日无分数的股票在新目标中权重为零。保留旧股票可能让每侧持股数超过 100。α 表示目标混合比例，不表示只使用 25% / 50% / 75% 的总资金，也不严格等于原始全量交易金额的对应比例。

### 混合规则：3 个配置

- `buffer_10_20_partial50`：先形成 10% 持仓 / 20% 保留带目标，再进行每日 50% 渐进调整。
- `buffer_10_20_every2_phase0`、`buffer_10_20_every2_phase1`：形成相同缓冲目标，每 2 个交易日调整一次，报告两个相位。

因此总数为 `1 + 4 + (2 + 3 + 5) + 3 + 1 + 2 = 21`。每个配置分别跑 `gross`、`fees_slip0`、`fees_slip2`、`fees_slip5`。后两档的买/卖总成本分别为 5/10bp、8/13bp，见 [四种成本定义](../README.md#4-种成本情景)。

## 结果怎么读

全 21 配置的基准成本结果和四成本年化对照在 [总览](../README.md)，完整分年、滑点和相位范围在 [原始报告](../results/20260917_turnover_trials/README.md)。本批 30% 保留带净年化 35.46%、净 Sharpe 1.991、换手 0.870；每侧再加 5bp 后净年化 13.49%。

年化为日收益均值 × 252，累计为复利净值减 1；2026 年只到 8 月。分年不重置仓位。末日缺开盘报价时无法成交，剩余头寸继续估值，金额见 `terminal_residual_value`。这是回顾性固定方案比较，不能据冠军结果称参数已在独立测试集验证。

## 服务器上保存在哪里

连接：`ssh -p 38527 root@connect.cqa1.seetacloud.com`。

| 内容 | 服务器绝对路径 |
| --- | --- |
| 本批运行根目录 | `/root/autodl-tmp/turnover-trials-20260917/` |
| 当次执行的源码 | `/root/autodl-tmp/turnover-trials-20260917/backtest/` |
| 最终复核运行结果 | `/root/autodl-tmp/turnover-trials-20260917/backtest/results/20260917_turnover_trials_verified/` |
| 原始分数与早期策略运行目录 | `/root/autodl-tmp/raw-score-backtest-20260916/` |
| 已复现的每日十分组基线 | `/root/autodl-tmp/raw-score-backtest-20260916/backtest/results/20260917_decile_strategy_preview/` |
| 原始模型分数 | `/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet` |
| 模型推理审计 | `/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/inference_audit.json` |
| 复权日开盘 / 收盘行情 | `/root/autodl-tmp/strategy-application-20260906/results/daily_prices.parquet` |

仓库内 `results/20260917_turnover_trials/` 收录的是 `_verified` 的最终复核结果，映射和哈希见 [collection.json](../results/20260917_turnover_trials/collection.json)。服务器历史运行目录采用原布局；当前仓库把源码、测试与协议整体迁到 `backtest/src/`。下列命令适用于**当前仓库根目录**，历史执行位置以上表为准。

## 在当前目录布局下复现

回测、数值测试和指标审计在 AutoDL 运行。本地只编辑代码、整理已存在的小型结果和核对哈希。环境依赖见 [requirements.txt](../requirements.txt)，代码自动核对输入、模型和协议哈希。

先运行现有的 14 项引擎测试和 8 项规则测试：

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
trial_python=/root/autodl-tmp/conda-envs/quant/bin/python
"$trial_python" -m unittest discover -s backtest/src -p 'test_*.py' -v
```

再运行固定的 84 组配置。`NEW_TURNOVER_RUN` 必须是尚不存在的输出目录：

```sh
trial_output=backtest/results/NEW_TURNOVER_RUN
trial_scores=/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet
trial_prices=/root/autodl-tmp/strategy-application-20260906/results/daily_prices.parquet
trial_inference=/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/inference_audit.json
trial_baseline=/root/autodl-tmp/raw-score-backtest-20260916/backtest/results/20260917_decile_strategy_preview
"$trial_python" backtest/src/run_turnover_trials.py \
  --raw-scores "$trial_scores" \
  --daily-prices "$trial_prices" \
  --inference-audit "$trial_inference" \
  --baseline-results "$trial_baseline" \
  --out "$trial_output"
"$trial_python" backtest/src/validate_turnover_trials.py "$trial_output"
"$trial_python" backtest/src/audit_turnover_candidate.py \
  --results "$trial_output" --raw-scores "$trial_scores" \
  --daily-prices "$trial_prices" --strategy buffer_10_30
"$trial_python" backtest/src/report_turnover_trials.py "$trial_output"
```

原始分数的三个历史基线可以用 `run_raw_scores.py` 的默认协议运行；纯多头与每日十分组使用显式协议：

```sh
"$trial_python" backtest/src/run_raw_scores.py \
  --raw-scores "$trial_scores" --daily-prices "$trial_prices" \
  --inference-audit "$trial_inference" \
  --protocol backtest/src/protocol_long_only.json \
  --out backtest/results/NEW_LONG_ONLY_RUN
"$trial_python" backtest/src/validate_results.py backtest/results/NEW_LONG_ONLY_RUN
```

每日十分组改用 `backtest/src/protocol_decile.json` 和另一新输出目录；协议只定义零费用与基准费用两种情景。这些较早批次不计入 84 组。

## 完成条件及归档关系

运行入口计算完只写 `state=computed`；独立验证通过才写 `state=complete`、`completed=84`、`total=84`、`validation=passed`。还须查看 `audit.json` 与候选审计 `candidate_audit.json`。当前归档包含：

- 84 组逐日费用、现金/持仓、收盘净值、隔夜/日内损益和指标核对；
- 168 组分年统计独立复算；
- 每日十分组与既有基线逐日净值一致；
- 30% 保留带 400 日目标持仓独立核对，未来输入扰动后前 201 日净值和持仓不变；
- `execution.json` 的运行主机、输入和代码哈希，以及 `collection.json` 的结果文件哈希。

此次目录整理移动源码、测试和协议，保持原始字节不变，因此 `execution.json` 中按文件名记录的代码哈希仍可在 `../src/` 找到对应文件。结果目录中的历史 README 也是经过哈希记录的原始报告；阅读当前布局下的操作命令应以本页为准。

2026-09-17 在 AutoDL 独立临时目录验证当前 `backtest/src/` 布局：14 项引擎测试和 8 项规则测试全部通过，两个回测入口的 `--help` 均可正常启动。该检查验证目录迁移后的导入和协议查找，不重跑或覆盖历史 84 组结果。
