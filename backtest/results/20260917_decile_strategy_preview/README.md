# 每日 Q10−Q1：扣费累计收益与回撤预览

用户指定把多空策略改为两端十分组，与中证 1000 一起展示累计收益及回撤；已纳入正式演示稿第 14 页“可交易性检验”。2026-09-17 在 AutoDL 从原始模型分数重跑；未训练新模型、未做中性化或搜索参数。

## 持仓与结果

每天在信号日的 1,000 只股票中，按原始分数从低到高分成十组。Q10 的 100 只等权做多，Q1 的 100 只等权做空，目标分别为扣费后净值的 +100% 与 −100%，不除以 2。并列分数按股票代码升序排组。收盘后出分数，次日开盘调仓，连续持有并计入隔夜收益；不加排名缓冲。

沿用原策略页费用：买入 3bp、卖出 8bp，额外滑点为零。2025-01-02 至 2026-08-28，共 402 个报告日；首日现金、次日开盘建仓、末日开盘退出，末日无未退出持仓。指标来自实际收盘净值。

| 序列 | 日均单边换手 | 年化收益 | Sharpe | 累计收益 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q10−Q1，扣费后 | 1.505 | 13.52% | 0.784 | 21.17% | -14.76% |
| 中证 1000，市场参照 | — | 21.04% | 0.837 | 32.91% | -22.94% |

策略不扣费年化为 55.35%，扣费后复合年增长率为 12.79%。主表年化采用日收益均值 × 252；Sharpe 采用样本标准差、零无风险利率。换手为每日买卖成交额之和除以调仓前开盘净值的两倍，再取全期均值，以数值表示。

策略最大回撤峰值日 2025-12-19，谷值日 2026-05-20，2026-07-15 恢复此前高点。指数最大回撤峰值日 2026-05-13，谷值日 2026-07-30，截至期末未恢复该高点。

## 两张图的口径

- 上图累计收益：`100 × (closing_nav − 1)`，源自自融资账户的复利净值。
- 下图回撤：`100 × (closing_nav / historical_max_nav − 1)`，历史高点包含初始净值 1。
- 中证 1000 是价格指数，首日收盘归一，不扣交易费，仅作市场参照，**不从策略收益中扣除**。策略和指数的首末持有时刻存在原有差异。
- 本图不是第 12 页未扣费的 O2C 分组算术累计。第 12 页未模拟连续持仓、隔夜和交易成本，不能复用其 43.91 个百分点作为本策略累计收益。
- 其余成交边界沿用 [回测说明](../../README.md)：理想化开盘参考价、可卖空假设，未模拟借券费、涨跌停排队和非线性冲击。本次为看过此时期后指定的回顾性策略预览。

## 复现与检查

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/autodl-tmp/conda-envs/quant/bin/python -m unittest discover -s backtest -p test_raw_backtest.py -v
/root/autodl-tmp/conda-envs/quant/bin/python backtest/run_raw_scores.py \
  --raw-scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --daily-prices /root/autodl-tmp/strategy-application-20260906/results/daily_prices.parquet \
  --inference-audit /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/inference_audit.json \
  --protocol backtest/protocol_decile.json --out backtest/results/NEW_DECILE_STRATEGY
/root/autodl-tmp/conda-envs/quant/bin/python backtest/validate_results.py backtest/results/NEW_DECILE_STRATEGY
/root/autodl-tmp/conda-envs/quant/bin/python backtest/build_decile_strategy_preview.py \
  --results backtest/results/NEW_DECILE_STRATEGY \
  --market backtest/results/20260917_csi1000_reference \
  --raw-scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet
```

14 项测试全部通过，包括十分组选择、并列排序、各边 100% 敞口、上一交易日信号、初末费用和未来输入不影响历史仓位。`audit.json` 独立复算两种费用情景的账本与指标；`preview_audit.json` 另外检查 402 个信号日的两端选股和两条序列共 804 个回撤值。

`status.json` 为 `complete` / `validation=passed` / `completed=2`。`preview_daily.csv` 保存绘图所需的日期、净值、日收益、累计、历史高点、回撤与换手；`preview_summary.json` 保存准确统计值。原始数据及数值运算留在 AutoDL，本地仅取回小型结果并生成图像预览。
