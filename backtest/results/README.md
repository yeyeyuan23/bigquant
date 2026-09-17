# 回测结果索引

[回测总览](../README.md) · [策略细则与复现](../docs/TURNOVER_TRIALS.md) · [源码索引](../src/README.md)。

先看 [20260917_turnover_trials](20260917_turnover_trials/README.md)：它是最新的 **21 个持仓配置 × 4 种成本情景**，84 组全部完成并通过审计。日期目录按原始批次保留；每个目录保存当次结果和证据，不能将不同批次行数累加成独立策略数。

## 所有批次

| 批次 | 目的与规模 | 结果 / 验证入口 |
| --- | --- | --- |
| [20260917_turnover_trials](20260917_turnover_trials/README.md) | 最新：十分组信号的缓冲、低频、渐进及混合，共 84 组 | `comparison.csv`、`summary.csv`、`calendar_years.csv`；`status.json` 为 84/84、验证 passed |
| [20260917_decile_strategy_preview](20260917_decile_strategy_preview/README.md) | 每日 Q10−Q1 持仓基线，零成本与基准费用，共 2 组；已由最新批次复现 | `primary.csv`、`preview_summary.json`、`preview_daily.csv`、`audit.json` |
| [20260916_raw_scores](20260916_raw_scores/README.md) | 历史三规则：每日五分位、20%/30% 缓冲、五日平均排名，共 12 组 | `primary.csv`、`summary.csv`、`audit.json` |
| [20260917_long_only_buffer](20260917_long_only_buffer/README.md) | 20%/30% 排名缓冲的纯多头版本，共 4 组 | `primary.csv`、`summary.csv`、`legacy_replay_audit.json` |
| [20260917_raw_deciles](20260917_raw_deciles/README.md) | 十分组的下一日开盘到收盘因子收益，401 日 | `decile_summary.csv`、`decile_daily_returns.csv`、`audit.json` |
| [20260917_raw_quintiles](20260917_raw_quintiles/README.md) | 历史五分位的下一日开盘到收盘因子收益，401 日 | `quintile_summary.csv`、`quintile_daily_returns.csv`、`audit.json` |
| [20260917_csi1000_reference](20260917_csi1000_reference/README.md) | 市场参照与历史股票池一致性，402 日 | `market_reference.csv`、`summary.json`、`audit.json` |
| [20260916_score_input_comparison](20260916_score_input_comparison/README.md) | 历史补充：历史处理流程 vs 原始分数，共 12 组；不计入 84 组换手实验 | `summary.csv`、`selection_overlap.csv`、`audit.json` |

因子分组收益不包含策略隔夜持仓和成交成本。市场指数只作参照；输入诊断比较的是两套完整输入处理流程，不是单独的中性化消融。

## 最新批次的文件怎么用

以下文件均在 `20260917_turnover_trials/` 下。

| 文件 | 内容 |
| --- | --- |
| [README.md](20260917_turnover_trials/README.md) | AutoDL 生成的原始结果报告，含完整基准成本、分年及滑点表 |
| [comparison.csv](20260917_turnover_trials/comparison.csv) | 21 行基准成本指标及对应毛年化、毛 Sharpe |
| [review_table.csv](20260917_turnover_trials/review_table.csv) | 21 行综合表，含名称、分年、滑点和相对基线的变化 |
| [summary.csv](20260917_turnover_trials/summary.csv) | 完整 84 行，含换手、算术年化、Sharpe、CAGR、累计、回撤、费用及期末剩余市值 |
| [calendar_years.csv](20260917_turnover_trials/calendar_years.csv) | 84 × 2 = 168 行分年指标；2026 年只到 8 月 |
| [phase_ranges.csv](20260917_turnover_trials/phase_ranges.csv) | 每 2/3/5 日及缓冲+隔日四个规则族的起点敏感性 |
| [net_nav.csv](20260917_turnover_trials/net_nav.csv) | 基准成本下全部 21 配置的逐日收盘净值、收益和换手 |
| [daily.csv.gz](20260917_turnover_trials/daily.csv.gz) | 84 组完整逐日账本，包含损益、交易金额、费用、现金与持仓 |
| [protocol.json](20260917_turnover_trials/protocol.json) | 当次固定策略、成本、输入哈希与执行口径 |
| [status.json](20260917_turnover_trials/status.json) / [audit.json](20260917_turnover_trials/audit.json) | 84/84 完成状态及账本、指标验证 |
| [candidate_audit.json](20260917_turnover_trials/candidate_audit.json) | 30% 保留带的 400 日持仓目标与未来输入扰动核验 |
| [tests.txt](20260917_turnover_trials/tests.txt) | 14 + 8 项 AutoDL 测试日志 |
| [execution.json](20260917_turnover_trials/execution.json) | 主机、时间、输入和源码哈希、输出哈希 |
| [collection.json](20260917_turnover_trials/collection.json) | 从服务器 `_verified` 目录收回的文件及哈希 |
| [report_audit.json](20260917_turnover_trials/report_audit.json) | 报告表格和原始 README 的生成记录 |
| [formatting_equivalence.json](20260917_turnover_trials/formatting_equivalence.json) | 当时 CI 语法清理前后，全部 84 组账本和表格一致 |

## 归档与当前代码的关系

既有结果文件保持原内容，尤其是已在审计中记录哈希的 README、CSV 和 JSON。此次新增的是导航说明；没有重新计算历史收益。

历史 README 中的代码路径是当时布局。当前源码、测试和固定协议集中于 [../src/](../src/README.md)，例如旧 `backtest/run_turnover_trials.py` 对应新 `backtest/src/run_turnover_trials.py`。源码文件内容未变，最新 `execution.json` 按文件名记录的哈希继续有效。现行复现命令见 [运行说明](../docs/TURNOVER_TRIALS.md)，不同历史批次的精确版本仍以各自 `execution.json` 和 Git 历史为准。
