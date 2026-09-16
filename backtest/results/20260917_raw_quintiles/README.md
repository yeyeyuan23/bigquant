# 原始模型分数五分位评价（第 12 页）

在 AutoDL 上直接使用固定模型权重输出的原始 `factor`。不缩尾、不标准化、不做行业或 Barra 风格中性化；收益标签也不做回归剔除。

| 分组 | 日均实际 O2C 收益（bp） | 日均相对收益（bp） | 累计相对收益（百分点） |
| --- | ---: | ---: | ---: |
| Q1 最低分 | 9.0072 | -4.6609 | -18.6902 |
| Q2 | 12.5433 | -1.1248 | -4.5106 |
| Q3 | 15.1484 | 1.4803 | 5.9359 |
| Q4 | 15.6745 | 2.0064 | 8.0456 |
| Q5 最高分 | 15.9672 | 2.2991 | 9.2192 |

Q5−Q1 日均差为 **6.95995bp（展示 6.96bp）**，五组日均收益依次上升。累计曲线是各日相对收益的算术和，并非可交易组合的复合净值。

## 时间与分组

- 401 个信号日：2025-01-02 至 2026-08-27；对应实际收益日为 2025-01-03 至 2026-08-28。
- 每个信号日有 1,000 个原始分数，先按分数分为五组，每组 200 只；并列分数按股票代码稳定排序。
- 分组在合并未来收益标签之前完成。401 日共有 434 个股票日缺少有效次日收益，从原组均值中排除，不重新分组、不补入其他股票；最终有效收益记录 400,566 条。
- 每组先计算真实次日 `close/open−1` 的等权均值，再减当天五组收益均值。这个基准是五组等权平均，不是外部指数，也不是行业或风格中性化。
- 不扣费用，不模拟持仓执行，不计隔夜；与第 16 页连续持仓回测回答不同问题。

旧第 12 页使用处理后分数，并在筛掉缺失标签后分组，Q5−Q1 为 8.63bp。本次改为原始分数，并将分组提前到读取未来收益之前。因此两数的差不能严格全部归因于中性化这一个步骤；旧文件保留为历史来源。

## 复现与验证

在 AutoDL 的仓库根目录执行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/quant/bin/python backtest/evaluate_raw_quintiles.py \
  --scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --labels /root/bigquant_private_data/private_o2c_labels_20250101_20260828.parquet \
  --out backtest/results/NEW_RAW_QUINTILES
```

脚本独立用 NumPy 排序与逐组求均值，逐日核对 Pandas 结果，再从五组收益矩阵独立复核均值、去均值和累计曲线。标签以 float64 累加，原始存储值保持不变。全部通过后写入 `audit.json` 和 `status.json`（`state=complete`）。代码与输入、结果哈希及主机记录在审计文件中。

本次运行目录：`/root/autodl-tmp/raw-score-backtest-20260916/backtest/results/20260917_raw_quintiles`。本机只编写代码、取回结果与排版。
