# 数据合同

本文件是取数、时间对齐和落盘格式的唯一事实来源。历史查询耗时、一次性短窗
日志和候选表现不在此保存。

## 环境边界

| 环境 | 职责 |
|---|---|
| BigQuant AIStudio | 查询比赛真实数据、完成字段与 PIT 核验、把分钟数据聚合为日频共享面板 |
| 本地 `quant` 环境 | 候选实现、单因子评价、组合训练、报告和冻结 |
| 比赛网页 | 提交冻结 Notebook、查看分数和排名 |

本地检查只能证明代码和接口正确，不能替代 AIStudio 真实数据结论。

## 官方数据表

| 对象 | 表名 | 主键/粒度 | 关键规则 |
|---|---|---|---|
| 比赛股票池 | `bigalpha_2026_instruments` | `date, instrument`，每日 1000 只 | 所有研究面板的左表 |
| 分钟行情与盘口 | `bigalpha_2026_stock_bar1m` | 股票分钟 | `volume、amount、deal_number` 是分钟增量 |
| 财务披露 | `bigalpha_2026_financial` | 披露事件 | `date` 是可见日，`report_date` 是报告期 |
| 风险暴露 | `bigalpha_2026_exposure` | 股票日 | 仅用于分层、中性化和增量诊断 |
| 公共日线 | `cn_stock_bar1d` | 股票日 | PV 的优先数据源 |
| 交易日历 | `all_trading_days` | 市场日 | 使用 `market_code='CN'` |
| 公开因子库 | `bigalpha_2026_factorlib` | 股票日 | 36 个公开特征，只用于重复性与增量评价 |

## 通用连接规则

- 必须以比赛股票池为左表；分钟表不能反推股票池。
- 池内停牌股票保留键，特征或标签允许缺失，不能跳到下一次交易日。
- 下一期标签先由中国交易日历得到 `next_date`，再连接对应收益；禁止按单股
  直接 `shift(-1)`。
- DAI 查询必须带日期或证券过滤，只投影必要字段，优先在 DAI 侧聚合。
- 禁止把标签、未来收益、公开评分或候选输出写入共享特征面板。

## 四类共享面板

### PV：日频量价

- 源表：`cn_stock_bar1d`
- 主键：`date, instrument`
- 可用时点：当日收盘后
- 列：

```text
date, instrument, open, high, low, close, pre_close,
amount, volume, deal_number
```

### HF：分钟成交的日级组件

- 源表：`bigalpha_2026_stock_bar1m`
- 主键：`date, instrument`
- 可用时点：当日收盘后
- 上午与下午分别计算分钟收益，禁止跨午休连接。
- 列：

```text
date, instrument,
minute_count, total_amount, total_volume, total_deal_number,
net_log_return, absolute_log_return,
tail_60_amount, tail_60_deal_number,
avg_trade_value, avg_trade_volume,
directional_efficiency, tail_trade_value_ratio,
shock_q90_active_count, shock_q90_mean_abs_return,
shock_q90_recovery_5m_median
```

### OB：分钟盘口的日级组件

- 源表：`bigalpha_2026_stock_bar1m` 五档字段
- 主键：`date, instrument`
- 可用时点：当日收盘后
- 只有价格和数量均为正的档位才有效。
- 快照变化不能直接解释为新增委托或撤单。
- 列：

```text
date, instrument,
minute_count, valid_snapshot_count,
both_sides_valid_rate, full_five_levels_rate,
tail_60_valid_best_quote_minutes,
tail_60_relative_spread_median,
tail_60_depth_completeness_median,
tail_60_bid_depth_imbalance_median,
negative_mid_shock_q10_bid_depth_recovery_5m_median,
full_day_depth_shape_median,
tail_60_depth_shape_median,
tail_60_shape_sign_consistency
```

### FR：财务披露事件

- 源表：`bigalpha_2026_financial`
- 唯一键：

```text
disclosure_date, instrument, report_date, category, shift
```

- `disclosure_date`：市场可见日。
- `effective_date`：披露日后的首个中国交易日。
- `report_date`：财务报告期，不用于市场可见时间对齐。
- `shift=0`：当时最新报告期。
- `total_assets` 使用 `lf`；流量字段必须显式选择 `lf、mrq` 或 `ttm`。
- 日级使用时，对 `effective_date` 做 backward as-of，不允许未来披露回填。

## 公开基础因子库

- 字段合同在 `src/bigalpha2026/factorlib.py`。
- 本地推荐目录：

```text
data/features/FACTORLIB/year=YYYY/part-YYYY.parquet
```

- 必须验证严格列集合、空主键、重复键、股票池缺失、有限值和覆盖率。
- 只用于：
  - 与公开因子的 Rank 相关性；
  - 残差信息检查；
  - 基础模型与“基础模型 + 候选”的滚动正则增量比较。

## 本地目录

```text
data/
├── universe/
├── features/
│   ├── PV/
│   ├── HF/
│   ├── OB/
│   ├── FR/
│   └── FACTORLIB/
├── exposures/
├── labels/
└── factors/
```

- 数据按年份分区；单个年度仍过大时才按月拆分。
- 日频数据主键为 `date, instrument`。
- 因子长表主键为
  `date, instrument, candidate_id, factor_version`。
- 每个数据集记录 `source_frequency、panel_frequency、available_time、
  date_role、schema_version、generated_at`。

## 候选输出合同

候选实现必须返回：

```text
date, instrument, factor
```

并满足：

- 股票日期无重复；
- 日期与证券代码已规范化；
- 因子值有限；
- 因子值越大代表预期收益越高；
- 股票池键完整，停牌缺失按登记规则处理中性填充或保留缺失；
- 修改未来输入不会改变过去输出。

提交 Notebook 的 `main(datasources, start_date, end_date)` 还必须：

- 接受平台传入的任意日期范围；
- 不依赖本地文件或外部网络；
- 不写死研究区间；
- 只返回一个因子；
- 保持官方模板的数据源包装和三列输出。
