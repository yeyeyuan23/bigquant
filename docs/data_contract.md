# 数据合同与 AIStudio 平台验证

本清单不绑定任何具体候选。每增加一类数据或一个候选，都必须重新执行相关项目。

## Step 0 v2 重新核验结果

核验日期：2026-07-25。真实查询在网页 AIStudio 执行，核验入口为 `/home/aiuser/work/factor_research_minimal.ipynb`，机器可读结果为 `/home/aiuser/work/step0_schema.json` 和 `/home/aiuser/work/step0_validation.json`。本轮没有生成或评价因子。

### 表与字段

| 数据对象 | 表 | 重新确认的字段 |
|---|---|---|
| 历史股票池 | `bigalpha_2026_instruments` | `date、instrument、name` |
| 分钟成交与盘口 | `bigalpha_2026_stock_bar1m` | 分钟 OHLC、`pre_close、adjust_factor、amount、volume、deal_number`；五档 `bid/ask_price1..5、bid/ask_volume1..5、bid/ask_num_orders1..5` |
| 财务披露 | `bigalpha_2026_financial` | `date、instrument、report_date、shift、category` 及财务字段；本轮重点复核 `total_assets、net_profit、net_cffoa、operating_revenue` |
| 风险暴露 | `bigalpha_2026_exposure` | 十个风格暴露、`industry_level1_code、float_market_cap、weights、ret` 及行业哑变量 |
| 公共日线 | `cn_stock_bar1d` | OHLC、成交量额笔数、换手、涨跌停价 |
| 中国交易日历 | `all_trading_days` | `date、market_code` |

### 历史股票池

2019—2024 的逐年结果如下：

| 年份 | 交易日 | 每日最少 | 每日最多 | 重复 `date、instrument` |
|---:|---:|---:|---:|---:|
| 2019 | 244 | 1000 | 1000 | 0 |
| 2020 | 243 | 1000 | 1000 | 0 |
| 2021 | 243 | 1000 | 1000 | 0 |
| 2022 | 242 | 1000 | 1000 | 0 |
| 2023 | 242 | 1000 | 1000 | 0 |
| 2024 | 242 | 1000 | 1000 | 0 |

正式面板继续以该表为左表。不能从有分钟行情的证券反推比赛股票池。

### 分钟口径

2024-01-02 的 1000 只池内股票中，997 只有分钟行情，另外 3 只保留为池内停牌或无行情记录。997 只活跃股票均为 240 行，最早 09:31、最晚 15:00，11:30 后至 13:01 前记录数为 0。

抽取 20 只股票比较分钟和日线：

- `SUM(volume) / daily.volume = 1.0000000000`；
- `SUM(amount) / daily.amount = 0.9999999988`；
- `SUM(deal_number) / daily.deal_number = 1.0000000000`；
- 最大单分钟值分别只占全日成交量、成交额和成交笔数约 4.96%、5.00% 和 3.36%。

因此 `volume、amount、deal_number` 是分钟增量，按日直接求和，不做差分。

### 盘口异常

2024-01-02 池内 997 只活跃股票共 239,280 个分钟快照：

- 双侧至少各一档价格和数量均为正：99.757606%；
- 双侧完整五档价格和数量均为正：98.917586%；
- 买一或卖一价格/数量无效：580 个快照；
- 有效买一价高于有效卖一价：0 个快照。

零价或零量继续解释为该档无有效报价。盘口组件必须逐档动态筛选并输出有效档数；不得把快照差分解释为新增委托或撤单。

### 财务 PIT

2023-12-01—2024-02-15 的 `shift=0` 短窗中，`lf、mrq、ttm` 各 41 行：

- `total_assets`：`lf` 41/41，`mrq` 和 `ttm` 均为 0/41；
- `net_profit`：`lf、mrq` 41/41，`ttm` 39/41；
- `net_cffoa`：`lf、mrq` 39/41，`ttm` 37/41；
- `operating_revenue`：`lf、mrq` 41/41，`ttm` 40/41。

周末披露样本再次确认 `date` 是信息可见日，`report_date` 是报告期。例如 2023-01-21 的披露在春节后的 2023-01-30 才进入交易日状态；2023-02-11 的披露在 2023-02-13 生效。未知日内披露时间时，继续保守使用严格晚于披露日的首个中国交易日。

### 风险暴露与标签

2024-01-02 风险表与历史股票池连接后，`SIZE、LIQUIDTY、industry_level1_code、float_market_cap` 均为 1000/1000 非空。

标签使用显式中国交易日历自连接确定下一交易日。2024-01-02 至 2024-01-05 每日股票池均为 1000 只，下一交易日日线缺失分别为 3、3、2、2 只。缺失保留为缺失，不允许按单股 `shift(-1)` 跳到下一次有交易的日期。

### 查询审计

- 首次字段探测未带 `filters`，平台按预期拒绝全表扫描；修正后所有表均使用日期分区。
- 单日分钟查询若把 `filters` 起止写成同一天会只覆盖午夜，表现为分钟结果 0 行；固定使用“起始日、次日”的半开分区范围。
- `strftime` 聚合字符串触发过 Arrow 字典类型导入异常；改为输出原生时间戳，财务分类显式转为字符串后通过。
- 这些失败日志保留在 AIStudio，用于防止以后重复相同错误。

## Step 2 共享面板最小分区验收

核验日期：2026-07-25。真实数据聚合、结果查看和 Parquet 读回均在 `/home/aiuser/work/factor_research_minimal.ipynb` 完成。本轮只生成合同探针，不计算因子、标签、IC、分组收益或多年结果。

| 数据族 | 最小范围 | 输出 | 关键结果 |
|---|---|---|---|
| `PV` | 2022-05 | `/home/aiuser/work/features/PV/pv_contract_probe_2022_05.parquet` | 19,000 行、10 列、19 个交易日；每日严格 1,000 只；零重复；活跃日线覆盖 99.8105%；读回一致 |
| `HF` | 2024-01-02 | `/home/aiuser/work/features/HF/hf_contract_probe_2024_01_02.parquet` | 1,000 行、17 列；239,280 条分钟记录；997 只活跃股票均 240 分钟；全部组件覆盖 99.7%；零重复；读回一致 |
| `OB` | 2024-01-02 | `/home/aiuser/work/features/OB/ob_contract_probe_2024_01_02.parquet` | 1,000 行、14 列；239,280 条快照；双侧有效率 99.757606%，五档完整率 98.917586%；全部组件覆盖 99.7%；零重复；读回一致 |
| `FR` | 2022-04—2022-05 | `/home/aiuser/work/features/FR/fr_contract_probe_2022_04_05.parquet` | 同期历史池并集 1,000 只；全市场 5,600 条事件过滤为 2,592 条池内事件、999 只实际披露股票；零重复；生效日严格晚于披露日；读回一致 |

结论：

- `PV` 直接使用 `cn_stock_bar1d`，不再为了日线字段扫描分钟表。
- `HF、OB` 各自一次读取必要分钟字段，同时产出本族两个候选需要的日级通用组件；不为候选编号重复查询。
- `FR` 保留事件粒度，但必须先按目标区间历史股票池证券并集过滤，不能把全市场财务事件下载到本机。
- 池内停牌或无行情股票在 `PV、HF、OB` 日级面板中保留主键并允许组件为空。
- OB 首次续跑时，尾盘方向一致性的标量写法由 `values.mean().abs()` 修正为 `abs(values.mean())`；修正只重跑聚合后半段，没有再次扫描分钟表。
- OB 输出中的 56.466 秒包含交互式诊断停顿，不是可比较的正式运行耗时；正式分区任务必须单独计时。
- 四个探针均已证明存储边界可运行；后续正式生成结果见下文，数据验收不代表任何候选有效。

## 2019—2023 低成本正式基础包

生成日期：2026-07-25。AIStudio 入口仍为 `/home/aiuser/work/factor_research_minimal.ipynb`。本轮只生成和下载正式数据，没有计算候选、IC、分组收益或组合结果。2019 单年先行验收后，2020—2023 沿用同一冻结合同逐年生成。

| 年份 | 交易日 / 行数 | PV 收盘覆盖率 | 风险暴露覆盖率 | 主标签覆盖率 |
|---:|---:|---:|---:|---:|
| 2019 | 244 / 244,000 | 99.777869% | 99.804918% | 99.777049% |
| 2020 | 243 / 243,000 | 99.811934% | 99.900000% | 99.811523% |
| 2021 | 243 / 243,000 | 99.862140% | 99.900000% | 99.861728% |
| 2022 | 242 / 242,000 | 99.897521% | 99.900000% | 99.897934% |
| 2023 | 242 / 242,000 | 99.941322% | 99.900000% | 99.939669% |

每年固定包含 `data/universe/year=YYYY/part-YYYY.parquet`、`data/features/PV/year=YYYY/part-YYYY.parquet`、`data/exposures/year=YYYY/part-YYYY.parquet` 和 `data/labels/year=YYYY/part-YYYY.parquet`。`data/manifest_YYYY.json` 记录列、形状、覆盖率、SHA-256、生成时间和 `evaluation_performed=false`。

下载后在本地 `conda quant` 环境逐文件核对 SHA-256、列、形状、重复键、每日行数、PV 特征合同和跨表键；每年四张表完全对齐，均为零重复且每日严格 1,000 只。标签继续使用精确下一交易日，停牌缺失不向后跳日。传输用临时 zip 在本地验收后删除；AIStudio 保留 `/home/aiuser/work/research_data` 下的正式 Parquet 与 manifest。

## FR 正式 PIT 事件面板

生成日期：2026-07-25。查询窗口从 2017-01-01 开始，平台实际返回 2018—2022 的披露；证券范围固定为 2019—2022 历史股票池并集。面板共 61,500 行、1,764 只股票，其中 `lf` 与 `ttm` 各 30,750 行，零重复键，10,888 行为周末披露。所有 `effective_date` 都是严格晚于 `disclosure_date` 的首个中国交易日。

本地文件位于 `data/features/FR/year=2018` 至 `year=2022`，统一清单为 `data/manifest_FR_2017_2022.json`。在 2019 年首个交易日的 1,000 只股票中，1,000 只均有至少一条此前可用的 LF 资产记录，且均有至少两条此前可用的 TTM 记录；因此 FR-001/002 的首次差分状态具备完整历史起点。该检查只验证 PIT 数据充分性，`evaluation_performed=false`。

2023 样本外事件另存为 `data/features/FR/year=2023/part-2023.parquet` 和 `data/manifest_FR_2023.json`：8,616 行、1,202 只股票，LF/TTM 各 4,308 行，2,090 行为周末披露，零重复；所有生效日严格晚于披露日。它只补充冻结后的 2023 状态，不改写开发期事件。

## HF/OB 正式日频共享面板

生成日期：2026-07-25。AIStudio 按自然月各扫描一次分钟表，同时产出 HF 和 OB 两族通用日频组件；本机不保存原始分钟或盘口数据。默认首轮固定使用 2019—2023 每年 2、8 月，共 10 个必跑月份。5、11 月构成预先声明的备选池，最多只允许按市场状态覆盖缺口启用 6 个月，禁止根据候选 IC 或收益挑月。

本地现有 18 个月：10 个必跑月份全部齐备；另有 2019—2022 年 5、11 月共 8 个备选档案，2023 年备选月未生成。因子值读取前，市场收益、截面波动和流动性检查发现必跑开发样本未覆盖低流动性月度尾部，因此唯一启用 `2022-11`；其余备选月仍不参与评价。每月 HF/OB 主键完全一致、零重复、每日严格 1,000 只，manifest 均为 `evaluation_performed=false`。组件最低覆盖率区间为 HF 90.2188%—99.5733%、OB 89.8563%—99.4933%；最低点为 2022-02，故首轮技术评价必须报告原始组件覆盖与最终因子覆盖，不能用中性填充掩盖数据缺口。

2023-02、2023-08 的 HF/OB 主键与 2023 基础包股票池逐日完全对齐。两个月只作冻结规则下的样本外确认，不得用于改变公式、方向、门槛或备选月份选择。

首轮因子构造发现 8 个“月末本身为交易日”的日期组件全空，原因是原查询把自然月末同时当作开区间结束边界。已在 AIStudio 用“下一自然日为排他结束”逐日补齐并覆盖原月分区；8 天 HF/OB 有效股票覆盖率为 99.5%—99.9%，更新后的 manifest 重新记录 SHA-256、形状、列和 `month_end_repair`。以后所有月分区必须显式使用 `[month_start, next_month_start)`，并断言每个池内交易日至少存在非空原始组件，不能只检查 1,000 个左表键。

## 环境边界

原始表查询、字段核验、PIT/异常处理验证和数据族日频聚合在网页 AIStudio 完成。经平台工具聚合并通过官方 UI 下载的日频 Parquet，可在本地 `conda quant` 环境执行版本化评价。

本机执行：

- Python 语法和导入检查；
- 文件及 Notebook JSON 结构检查；
- `main()` 是否存在及返回列名检查；
- 使用不含比赛信息的最小虚拟输入检查明显接口错误；
- 读取固定日频 Parquet，运行因子生成、IC、分组、组合和消融；
- 检查提交代码中不存在本地路径、外部网络和写死日期。

本机不得扫描或保存原始分钟、盘口和财务全表；研究 Parquet 不得被提交 Notebook 依赖。最终提交仍必须能在 AIStudio 仅靠平台数据重建因子。

## 数据字段

- 表名和字段名与平台数据页一致；
- 日期、时间和证券代码类型已确认；
- 字段的单位、复权和累计口径已确认；
- 分钟累计字段能够正确处理跨日、午休、停牌和重置；
- 盘口档位和价格数量字段能够正确对应；
- 财务公告日、报告期、类别和偏移含义已确认；
- 历史股票池按当日成分获得；
- 标签只使用因子日之后的数据。

## PIT 与防泄漏

- 因子日只能使用当时已经披露或生成的数据；
- 财务记录在披露日前不可见；
- 所有滚动窗口只向历史方向取值；
- 修改未来行情不改变过去因子；
- 修改未来财务记录不改变过去因子；
- 模型的训练、验证和测试按日期隔离。

## 输出接口

- 结果列严格为 `date、instrument、factor`；
- 无重复股票日期；
- 无缺失交易日；
- 每日缺失率低于比赛上限；
- 内部覆盖率目标不低于 95%；
- 因子值全部有限；
- 每日具有足够截面差异；
- 结果按日期和股票排序。

## 资源与运行

1. 使用最小样本核验字段。
2. 使用短日期全股票池核验覆盖率。
3. 记录查询耗时、内存和输出规模。
4. 可复用的原始聚合只计算一次。
5. 完整历史使用满足内存要求的环境。
6. 优化 DAI 侧聚合和传输规模。
7. Notebook 完整运行不得超过 3 小时。
8. 私榜增量调用不得依赖本机、外部网络或预生成因子文件。

## 研究输出

AIStudio 保存数据合同、数据族聚合日志、查询性能和 Parquet 行数/覆盖率；本地保存候选登记表、单因子评价、组合和消融评价、年度及市场状态评价、相关矩阵、滚动 Elastic Net 权重、ModelScore 代理与冻结记录。两侧结果必须能通过脚本和校验和对应。

## 提交前

- 候选已经通过计划中的全部准入门槛；
- Notebook 只有一个因子入口；
- `main()` 可以使用平台传入的任意日期范围；
- 未写死历史评估区间；
- 未引用本地路径；
- 未使用外部网络；
- 未包含第二个 Notebook；
- 上传文件与冻结代码版本一致。

## 四类共享特征面板合同

版本：`v1`；冻结日期：2026-07-25。

共享面板只保存同一数据族可复用的可见信息和描述性组件，不保存候选编号绑定的原始因子、未来收益、标签、中性化结果或组合权重。

### 总体规则

- AIStudio 以历史股票池为左表生成日级研究面板；池内停牌股票保留键并允许特征缺失。
- `PV、HF、OB` 主键为 `date、instrument`，面板频率均为日频。
- `FR` 保留披露事件粒度，唯一键为 `disclosure_date、instrument、report_date、category、shift`。
- Parquet 按数据族和年份分区；单个年度分区过大时才继续按月拆分。
- 元数据记录 `schema_version、source_frequency、panel_frequency、available_time、date_role、generated_at`。
- 落盘前严格校验列集合、空主键和重复主键；标签只在评价时单独连接。

### PV：日频量价

| 项目 | 固定合同 |
|---|---|
| AIStudio 源表 | `cn_stock_bar1d` |
| 源频率 / 面板频率 | `1d / 1d` |
| 可用时点 / 主键 | 当日收盘后；`date、instrument` |
| 落盘列 | `date、instrument、open、high、low、close、pre_close、amount、volume、deal_number` |
| 当前使用者 | `PV-001、PV-002` |

PV 不再为了日线字段重复扫描分钟表。历史活动基线、收益推进、振幅、隔夜跳空与日内吸收均在本地计算，不落入共享面板。

### HF：分钟成交的日级组件

| 项目 | 固定合同 |
|---|---|
| AIStudio 源表 | `bigalpha_2026_stock_bar1m` 的 `close、amount、volume、deal_number` |
| 源频率 / 面板频率 | `1m / 1d` |
| 可用时点 / 主键 | 当日收盘后；`date、instrument` |
| 质量与活动 | `minute_count、total_amount、total_volume、total_deal_number` |
| 路径与尾盘 | `net_log_return、absolute_log_return、tail_60_amount、tail_60_deal_number` |
| 成交效率 | `avg_trade_value、avg_trade_volume、directional_efficiency、tail_trade_value_ratio` |
| 冲击统计 | `shock_q90_active_count、shock_q90_mean_abs_return、shock_q90_recovery_5m_median` |
| 当前使用者 | `HF-001、HF-002` |

上午和下午分别计算分钟收益，禁止把午休跨段变化当成一分钟收益。冲击固定为股票当日绝对分钟收益前 10% 且活动度不低于中位数，恢复窗口固定为 5 分钟。

### OB：分钟盘口的日级组件

| 项目 | 固定合同 |
|---|---|
| AIStudio 源表 | `bigalpha_2026_stock_bar1m` 的五档买卖价量 |
| 源频率 / 面板频率 | `1m_snapshot / 1d` |
| 可用时点 / 主键 | 当日收盘后；`date、instrument` |
| 质量 | `minute_count、valid_snapshot_count、both_sides_valid_rate、full_five_levels_rate、tail_60_valid_best_quote_minutes` |
| 成本与深度 | `tail_60_relative_spread_median、tail_60_depth_completeness_median、tail_60_bid_depth_imbalance_median` |
| 恢复 | `negative_mid_shock_q10_bid_depth_recovery_5m_median` |
| 形状 | `full_day_depth_shape_median、tail_60_depth_shape_median、tail_60_shape_sign_consistency` |
| 当前使用者 | `OB-001、OB-002` |

每档只有价格和数量同时为正时才有效。组件只描述快照状态、深度结构与后续状态变化，不把盘口减少解释为撤单。

### FR：财务披露事件

| 项目 | 固定合同 |
|---|---|
| AIStudio 源表 | `bigalpha_2026_financial` 与 `all_trading_days` |
| 源频率 / 面板频率 | `event / event` |
| 可用时点 | 披露日后的首个中国交易日 |
| 唯一键 | `disclosure_date、instrument、report_date、category、shift` |
| 落盘列 | `disclosure_date、effective_date、instrument、report_date、category、shift、net_cffoa、net_profit、operating_revenue、total_assets` |
| 当前使用者 | `FR-001、FR-002` |

流量字段使用明确的 `ttm` 记录，`total_assets` 使用 `lf`。事件查询先限制为目标研究区间历史股票池的证券并集；本地再按每日历史股票池做 backward as-of 状态化。正式面板包含 2018 年披露历史，以支持 2019 年初状态和首次差分。

### 禁止落盘

- `pv001_raw、hf001_raw、ob002_raw` 等候选专用列；
- `factor、candidate_id、factor_version` 等因子输出列；
- `next_return、forward_return、label、target` 等未来信息；
- 中性化残差、截面排序、组合权重和评价结果。

机器可执行的列与主键校验位于 `src/bigalpha2026/feature_contracts.py`。
