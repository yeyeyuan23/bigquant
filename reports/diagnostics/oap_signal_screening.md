# Open Asset Pricing 信号第一轮筛选

## 1. 边界

本轮只使用 OAP 元数据和项目已有字段合同做**事前可实现性筛选**，没有读取任何
候选收益、IC、分组收益或公榜结果。筛选结果不是因子有效性结论，也没有把 OAP
美股信号值作为比赛输入。

- 元数据来源：
  `https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv`
- 获取日期：2026-07-26
- OAP 发布版本：v2.0.0（2025-10）
- 文件 SHA-256：
  `f6c055120bad7afe97c16e23e0f49bce0982269026605a955e8064bd661cf415`
- 项目字段来源：`docs/data_contract.md` 与
  `src/bigalpha2026/feature_contracts.py`

OAP 公式只作为文献机制来源。正式实现必须使用比赛指定数据，在 AIStudio
复核字段和 PIT 后运行；提交代码不能读取本地 OAP 文件或访问外部网络。

## 2. 信号总体结构

下载的 `SignalDoc.csv` 共 331 行：

| 类型 | 数量 |
|---|---:|
| Predictor | 212 |
| Placebo | 114 |
| Drop | 5 |

212 个 Predictor 的数据类别：

| OAP 数据类别 | 数量 | 当前处理 |
|---|---:|---|
| Accounting | 99 | 按现有 4 个 FR 数值字段逐项筛选 |
| Price | 45 | 优先筛选 |
| Trading | 13 | 优先筛选 |
| Analyst | 18 | 当前字段不可实现 |
| Other | 12 | 当前字段不可实现 |
| Options | 9 | 当前字段不可实现 |
| 13F | 8 | 当前字段不可实现 |
| Event | 8 | 当前字段不可实现 |

第一层因此排除 55 个依赖 Analyst、Other、Options、13F 或 Event 数据的预测
变量，剩余 157 个 Accounting、Price、Trading 信号进入字段级审查。

## 3. 筛选规则

按以下顺序判断，避免看到结果后修改方向：

1. **比赛数据可实现性**：只使用 PV、HF、OB、FR、exposure 和公开因子库合同
   已确认的字段。
2. **实现忠实度**：
   - `exact`：原定义核心字段和窗口均可重建；
   - `near_exact`：只替换市场基准或等价A股口径；
   - `adapted`：保留机制，但因字段差异修改财务或交易口径；
   - `proxy`：只能做机制代理，不能称为论文复现。
3. **PIT 与历史长度**：财务因子按 `effective_date` 生效；长窗口不能牺牲
   2019—2021 冻结开发期。
4. **公开库重复风险**：PE、PB、PS、ROA、Beta、五日动量、五日反转和五日
   波动率的简单变体不优先。
5. **目标周期**：OAP 的月度或 12 个月持有期只提供机制方向；BigAlpha 日频
   下一期增量必须重新验证。

## 4. 第一轮结论

机器可读清单见 `reports/diagnostics/oap_signal_shortlist.csv`。

| 层级 | 数量 | 含义 |
|---|---:|---|
| A | 15 | 当前字段可以直接或近似忠实实现；长窗口信号允许用 2019 年自然预热 |
| B | 11 | 已全部实现；保留原长窗口，字段不足者明确标为 adapted/proxy |
| C | 8 | 已被 FR-002 或公开因子库覆盖，不新增候选 |

### A层：建议先做

#### 财务与财务估值

1. `AssetGrowth`：总资产增长，方向按 OAP 为负。
2. `cfp`：经营现金流/流通市值，需与 PE、PB、盈利质量控制比较。
3. `EarningsSurprise`：改写为 PIT TTM 净利润变化惊喜。
4. `RevenueSurprise`：改写为 PIT TTM 营业收入变化惊喜。
5. `NumEarnIncrease`：改写为 TTM 净利润连续改善次数。

`EarningsSurprise`、`RevenueSurprise`、`NumEarnIncrease` 是
**OAP-inspired A股版本**，不能标为原论文精确复现。

#### 日频价格与交易

1. `MaxRet`：过去一个月最大单日收益。
2. `ReturnSkew`：过去一个月日收益偏度。
3. `High52`：收盘价相对过去 52 周最高价。
4. `Illiquidity`：绝对收益/成交额。
5. `BidAskSpread`：Corwin-Schultz 高低价价差估计。
6. `PriceDelayRsq`：个股对当期和滞后市场收益的反应延迟。
7. `CoskewACX`：个股收益与市场平方收益的协偏度。
8. `zerotrade1M`：股票池交易日中的零成交状态。
9. `IntMom`：`t-12` 到 `t-6` 月中期动量。
10. `IndMom`：一级行业内大盘或行业组合的滞后收益。

## 5. 首批实现批次

以下 8 个不需要补价格前史的候选已完成本地计算实现、方向测试、PIT 测试和
防未来泄漏测试，并于 2026-07-26 按现行 S/I 规则在本地核验快照重跑：

| 顺序 | OAP 信号 | 数据族 | 原因 |
|---|---|---|---|
| 1 | AssetGrowth | FR | 单字段、PIT 清楚、与 FR-002 机制不同 |
| 2 | MaxRet | PV | 计算简单、与五日波动不完全相同 |
| 3 | ReturnSkew | PV | 独立的收益分布形状机制 |
| 4 | Illiquidity | PV | 交易冲击机制，现有字段完整 |
| 5 | BidAskSpread | PV | 无需新增盘口下载 |
| 6 | zerotrade1M | PV | 现有成交量、成交笔数和股票池即可 |
| 7 | cfp | FR | PIT经营现金流与当日流通市值的财务估值机制 |
| 8 | RevenueSurprise | FR | 现有 PIT 收入历史即可构造A股改写版 |

精确结果见统一输出 `reports/routes/factor_pool_incremental.csv` 和
`reports/factor_pool_admission.csv`。当前 `PV-003` 仅通过 S，进入规则复合；
`FR-004、FR-005` 仅通过 I，进入两个联合模型。`PV-007/zerotrade1M` 因 21 日
离散计数无法达到每日 50 个不同值的技术门槛而淘汰，其余本批候选未获准。

不能根据已查看结果在原编号上选择相反方向；需要研究反向或改写机制时必须使用
新编号。

## 6. 第二批实现与本地评价

不再补拉 2018 年价格数据。`High52、PriceDelayRsq、CoskewACX、IntMom、
IndMom` 使用完整原定窗口，以 2019 年作为自然预热期，从首次产生有效值的日期
开始评价；不缩短窗口，也不把预热期缺失判为技术失败。财务快照本地已有 2018
前史。

| 候选 | 信号 | S | I | 结论 |
|---|---|---:|---:|---|
| `FR-006` | EarningsSurprise | 未通过 | 通过 | 进入两个联合模型 |
| `FR-007` | NumEarnIncrease | 技术淘汰 | 不运行 | 日截面只有少量离散计数 |
| `PV-008` | High52 | 未通过 | 通过 | 进入两个联合模型 |
| `PV-009` | PriceDelayRsq adapted | 通过 | 通过 | 进入三条管线 |
| `PV-010` | CoskewACX | 未通过 | 未通过 | 淘汰 |
| `PV-011` | IntMom | 未通过 | 通过 | 进入两个联合模型 |
| `PV-012` | IndMom | 技术淘汰 | 不运行 | 一级行业截面不足 50 个不同值 |

正式组合读取当前完整获准池，不根据候选加入顺序生成不同模型。

## 7. B级实现与评价

B级11个已全部登记到 `FR-008—011`、`PV-013—019`。详细公式、方向和忠实度
见 `src/bigalpha2026/candidates/OAP_CANDIDATES.md`。

| 候选 | 信号 | 忠实度 | S | I | 路由 |
|---|---|---|---:|---:|---|
| `FR-008` | EarningsConsistency | adapted | 未通过 | 未通过 | 淘汰 |
| `FR-009` | MeanRankRevGrowth | near_exact | 未通过 | 未通过 | 淘汰 |
| `FR-010` | AbnormalAccruals | proxy | 未通过 | 未通过 | 淘汰 |
| `FR-011` | AM | exact | 未通过 | 未通过 | 淘汰 |
| `PV-013` | Mom12m | exact | 未通过 | 通过 | 两个联合模型 |
| `PV-014` | RealizedVol | near_exact | 通过 | 通过 | 三条管线 |
| `PV-015` | VolSD | near_exact | 未通过 | 未通过 | 淘汰 |
| `PV-016` | std_turn | near_exact | 未通过 | 未通过 | 淘汰 |
| `PV-017` | VolumeTrend | adapted | 未通过 | 未通过 | 淘汰 |
| `PV-018` | LRreversal | exact | 未通过 | 未通过 | 淘汰 |
| `PV-019` | ResidualMomentum | proxy | 未通过 | 通过 | 两个联合模型 |

`PV-014` 是本批唯一通过 S 的候选；`PV-013、PV-014、PV-019` 通过 I。
在全部批次按同一规则重新路由后，三条正式管线 2022/2023 Rank IC 为：
规则复合 `0.05580/0.05794`、Elastic Net `0.03910/0.02807`、LightGBM
`0.03067/0.02700`。三条管线均覆盖每个验证年的 242 个交易日；排序使用两个
验证年中较差的结果优先、两年均值次优。

## 8. 明确不做

- 不直接下载或上传 OAP 美股公司层面信号。
- 不把 OAP 原论文 t 值当作A股有效性证据。
- 不把季度 EPS 因子伪装成当前 TTM 字段的精确复现。
- 不重复实现 `EP、BM、SP、Beta、STreversal、DolVol` 等公开库已有信息。
- 不再扩展清单外需要60—120个月历史的新信号，例如 `BetaTailRisk`。
- 不在本轮生成 5/10/20/60 日窗口网格。
