# 中证 1000：市场参照与股票池核验

本序列只作图中的市场参照，不是新增持仓规则，不参与策略收益相减或超额计算。

## 股票池

[BigAlpha AI 因子挖掘官方说明](https://bigquant.com/square/competition/76ad3f56-ec2b-431a-890e-139a7f4bbcba)明确股票池为中证 1000 在历史相应时点的成分股。在 AutoDL 将原始模型分数的日期和股票键与两份 `bigalpha_2026_instruments` 数据逐行比对：402 日、每天 1,000 只，共 402,000 条，完全一致。全期出现过 1,307 只股票，未把当前成分股固定回填整个历史。

## 指数口径

- 代码为 `000852.SH`，中证 1000 **价格指数**，不是含分红再投资的全收益指数。
- 取新浪日行情，并用腾讯全部 402 日的开盘、收盘价交叉核对；最大差为 0.005 点，与显示精度差相符。来源 URL、下载时间和响应哈希见 `audit.json`。
- 2025-01-02 收盘归一为 1，至 2026-08-28，按各日收盘绘图，不扣交易费用。策略则次日开盘建仓、末日开盘处理退出，端点持有时刻有差别。
- 全期累计 **32.91%**，日收益均值 × 252 为 **21.04%**，复合年增长率 **19.52%**，零无风险利率 Sharpe **0.837**，最大回撤 **-22.94%**。这些是市场描述统计，不是策略净表现。
- 不给指数填写策略换手或扣费后收益；不将指数线与多空策略的直接差值称为 alpha。

## 复现与结果

`market_reference.csv` 保存对齐的 402 个指数收盘、开盘、归一净值和日收益；`summary.json` 保存描述统计。原始 HTTP 响应保存在 AutoDL 同运行目录的 `_source` 文件夹。

```sh
/root/autodl-tmp/conda-envs/quant/bin/python backtest/build_market_reference.py \
  --raw-scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --instruments /root/bigquant_private_data/bigalpha_2026_instruments_20250101_20260801.parquet \
  --instruments /root/bigquant_private_data/bigalpha_2026_instruments_20260802_20260828.parquet \
  --calendar backtest/results/20260917_long_only_buffer/net_nav.csv \
  --out backtest/results/NEW_MARKET_REFERENCE
```

脚本只允许在 Linux/AutoDL 执行，遇到缺交易日、两家价格差异超出 0.011 点或股票池键不一致即停止。输出完成要求 `status.json` 为 `complete`、`validation=passed`，且 `audit.json` 为 `passed`。
