# 按自然年统计的不扣费收益

沿用冻结分数、持仓规则、成交日期与估值方法，将买入费用、卖出费用和额外滑点全部设为零重新回测。2025 年末按收盘净值划分两段，持仓与历史排名连续。

| 规则 | 2025 年累计收益 | 2026 年至 8 月 28 日累计收益 |
| --- | ---: | ---: |
| 每日五分位 | +41.34% | +16.30% |
| 排名缓冲区 | +47.47% | +21.01% |
| 五日平均排名 | +36.04% | +15.86% |

对应的扣费后收益与换手见 [扣费情景](../20260916_calendar_year_returns/README.md)。演示稿换手取扣费情景，两种情景分别按各自净值调整持仓规模。收益均为实际累计值。

```sh
python experiments/finals_pre/e7_strategy_application/yearly_close_returns.py \
  --scenario gross \
  --out reports/dependencies/finals_pre/e7_strategy_application/20260916_calendar_year_gross
```

输入默认读取本地 `data/runtime/finals_pre/e7_strategy_application/20260916/inputs/`。`audit.json` 保存输入、代码及输出哈希，并检查原持有区间复现、年度复合收益与零费用。
