# 纯多头排名缓冲：AutoDL 回测

在已有原始分数回测上只增加一个纯多头方案：前一交易日收盘分数用于次日开盘调整；优先保留仍处于前 30% 的旧持仓，再按分数补足股票池的 20%，每日等权。目标多头为扣费后净值的 100%，无空头、不借入现金。没有重新训练或搜索缓冲参数。

统计期为 2025-01-02 至 2026-08-28，402 日。首日现金、1 月 3 日开盘首次建仓，最后一日开盘处理退出。连续持有并计入隔夜。缺开盘报价时保留原股数；被锁定的旧仓继续占用资金，其他买入按可用资金缩减，不能因此借钱。末日仍有少量无法退出的持仓，继续按当时已知价格估值，不能称为全部变现。

| 情景 | 日均单边换手 | 算术年化收益 | Sharpe | 全期累计 | 复合年增长率 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 零费用 | 0.450 | 31.87% | 1.728 | 61.72% | 35.16% | -11.94% |
| 买入 3bp、卖出 8bp | 0.449 | 19.39% | 1.053 | 32.55% | 19.32% | -15.88% |
| 基准费用，每侧另加 2bp | 0.449 | 14.85% | 0.807 | 23.31% | 14.03% | -17.40% |
| 基准费用，每侧另加 5bp | 0.449 | 8.05% | 0.438 | 10.63% | 6.54% | -19.63% |

年化为日收益均值 × 252；Sharpe 使用样本标准差，乘 √252，无风险利率为零。换手为每日买卖金额合计 / 调仓前净值 / 2，再对 402 日取平均。纯多头与原多空组合敞口不同，不将结果差解释为纯粹的规则增益。

中证 1000 仅作为[市场参照](../20260917_csi1000_reference/README.md)，不从上述收益中扣除。全期指数累计 +32.91%，本方案净累计 +32.55%，不能宣称净收益明显跑赢指数。仍采用理想化开盘参考价、比例费用和复权价近似，未模拟涨跌停排队、容量、每笔最低佣金等。方案是在看过同段历史结果后新增的回顾性检验。

## 复现

在 AutoDL 仓库根目录执行：

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/autodl-tmp/conda-envs/quant/bin/python -m unittest discover -s backtest -p test_raw_backtest.py -v
/root/autodl-tmp/conda-envs/quant/bin/python backtest/run_raw_scores.py \
  --protocol backtest/protocol_long_only.json \
  --raw-scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --daily-prices /root/autodl-tmp/strategy-application-20260906/results/daily_prices.parquet \
  --inference-audit /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/inference_audit.json \
  --out backtest/results/NEW_LONG_ONLY_RUN
/root/autodl-tmp/conda-envs/quant/bin/python backtest/validate_results.py backtest/results/NEW_LONG_ONLY_RUN
```

12 项会计与时序测试通过。独立从逐日账本重算费用、现金与持仓净值、收益、换手和所有展示指标，并逐日确认无空头、无现金借款。原每日五分位和多空缓冲的毛/净四组结果也已重放，与旧账本数值差小于 1e-12，见 `legacy_replay_audit.json`。完成标志为 `status.json` 中 `state=complete`、`completed=4`、`validation=passed`。主机、输入和代码哈希见 `execution.json`。
