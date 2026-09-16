# 原始分数十分组：第 12 页

在 AutoDL 从原始模型分数重新计算，每个信号日 1,000 只股票按分数由低到高分为 Q1–Q10，每组 100 只。并列分数按股票代码稳定排序。先分组、再合并未来收益；不缩尾、不标准化、不做行业或风格中性化。

401 个信号日为 2025-01-02 至 2026-08-27，对应实际收益日为 2025-01-03 至 2026-08-28。收益为次日开盘至收盘的真实 `close/open−1`。434 个缺失收益记录仅在各自原组内排除，不重新分组；有效股票日 400,566 条。

每组等权平均收益减去当天十组收益的等权平均，得到相对共同基准的超额收益。累计是按日算术和，单位为百分点，没有复利、隔夜或成本。

| 组 | 日均实际收益（bp） | 日均超额（bp） |
| --- | ---: | ---: |
| Q1 | 4.92 | -8.75 |
| Q2 | 13.09 | -0.58 |
| Q3 | 12.59 | -1.07 |
| Q4 | 12.49 | -1.18 |
| Q5 | 15.10 | +1.43 |
| Q6 | 15.21 | +1.54 |
| Q7 | 15.51 | +1.84 |
| Q8 | 15.84 | +2.17 |
| Q9 | 16.06 | +2.39 |
| Q10 | 15.87 | +2.21 |

右图只展示 Q1、Q5、Q10 的累计相对收益与多空 Q10−Q1。期末依次为 **-35.07、+5.72、+8.84、+43.91 个百分点**。多空每日直接用 Q10 收益减 Q1 收益（+100% Q10、-100% Q1，不除以 2），两组的共同基准自然抵消。日均多空差为 **10.95105576bp**。这是分组收益检验，不是第 16 页扣费后的连续持仓净值。

十组均值**不严格单调**，Q2–Q4 与 Q9–Q10 存在倒序，最明显的区分在低分端。不能复用五分位版本的“各组依次上升”结论，也不能将更集中的两端收益差增大解释为模型改善。

## 复现与审计

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/quant/bin/python backtest/evaluate_raw_deciles.py \
  --scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --labels /root/bigquant_private_data/private_o2c_labels_20250101_20260828.parquet \
  --out backtest/results/NEW_RAW_DECILES
```

脚本逐日用独立 NumPy 排序与十等分复核全部 4,010 个分组收益，再独立检查去均值、累计与多空恒等式。`audit.json` 保存代码、输入和输出哈希；`status.json` 必须为 `state=complete`、`validation=passed`、`days=401`、`groups=10`。小型图表数据分别为 `decile_summary.csv`、`decile_daily_returns.csv` 与 `display_daily.csv`。本机只写代码、取回结果并排版。
