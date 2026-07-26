# 因子候选登记表

本文件只保存候选的当前定义、方向、状态和关键结论。数据字段口径见
`data_contract.md`，评价流程与门槛见 `factor_research_plan.md`。

## 登记规则

- 基础因子只有四类：`PV、HF、OB、FR`。
- 跨类组合不是第五类，统一登记为 `composite`，并显式列出 `data_sources`。
- 先登记、后写代码；看过结果后不得在原编号上翻转方向或改变机制。
- `data_checked` 只表示数据可实现，不表示因子有效。
- 所有最终因子均输出 `date、instrument、factor`，且值越大代表预期收益越高。

状态流转：

```text
registered
→ data_checked
→ implemented
→ single_tested
→ combination_tested
→ validated
→ frozen
→ submitted
```

失败候选标记为 `rejected` 并保留原因，不删除历史记录。

## 当前候选

| ID | 类别 | 机制 | 方向 | 周期 | 当前状态 |
|---|---|---|---|---|---|
| `PV-001` | PV | 活动强度相对价格推进效率 | 推进越有效越高 | 1—3 日 | `rejected` |
| `PV-002` | PV | 隔夜冲击的日内吸收 | 负跳空回补为正，正跳空回落为负 | 1—3 日 | `rejected` |
| `HF-001` | HF | 分钟价格冲击后的吸收与恢复 | 恢复越充分越高 | 1—3 日 | `rejected` |
| `HF-002` | HF | 成交碎片化条件下的价格效率 | 价格效率越高越高 | 1—3 日 | `rejected` |
| `OB-001` | OB | 有效档位盘口韧性 | 价差更低、深度恢复更强越高 | 1 日 | `combination_tested` |
| `OB-002` | OB | 盘口深度形状的持续偏斜 | 买方近端深度占优越高 | 1 日 | `rejected` |
| `FR-001` | FR | 新披露现金转化质量改善 | 现金转化改善越强越高 | 5—20 日 | `rejected` |
| `FR-002` | FR | 新披露资产效率改善 | 资产周转与 ROA 改善越强越高 | 5—20 日 | `combination_tested` |
| `PV-003` | PV | OAP 过去月度最大单日收益 | 最大单日收益越低越高 | 1—20 日 | `combination_tested` |
| `PV-004` | PV | OAP 日收益偏度 | 偏度越低越高 | 1—20 日 | `rejected` |
| `PV-005` | PV | OAP Amihud 非流动性 | 单位成交额冲击越大越高 | 1—20 日 | `rejected` |
| `PV-006` | PV | OAP Corwin-Schultz 价差估计 | 估计价差越宽越高 | 1—20 日 | `rejected` |
| `PV-007` | PV | OAP 零成交状态 | 零成交日占比越高越高 | 1—20 日 | `rejected` |
| `FR-003` | FR | OAP 总资产增长 | 总资产同比增长越低越高 | 5—20 日 | `rejected` |
| `FR-004` | FR | OAP-inspired TTM 收入增长惊喜 | 收入同比增长越高越高 | 5—20 日 | `combination_tested` |
| `FR-005` | FR | OAP 经营现金流市值比 | 经营现金流/流通市值越高越高 | 5—20 日 | `combination_tested` |
| `FR-006` | FR | OAP-inspired TTM 净利润增长惊喜 | 净利润同比改善越高越高 | 5—20 日 | `combination_tested` |
| `FR-007` | FR | OAP-inspired 连续盈利改善次数 | 连续改善次数越多越高 | 5—20 日 | `rejected` |
| `PV-008` | PV | 52周高点接近度 | 越接近过去52周高点越高 | 1—20 日 | `combination_tested` |
| `PV-009` | PV | 当期与滞后市场反应延迟 | 价格延迟越低越高 | 1—20 日 | `combination_tested` |
| `PV-010` | PV | 市场协偏度 | 协偏度越低越高 | 1—20 日 | `rejected` |
| `PV-011` | PV | 12至6个月中期动量 | 中期累计收益越高越高 | 1—20 日 | `combination_tested` |
| `PV-012` | PV | 一级行业动量 | 所属行业过去收益越高越高 | 1—20 日 | `rejected` |
| `FR-008` | FR | 48个月盈利增长一致性 | 平均盈利增长越高越高 | 5—20 日 | `rejected` |
| `FR-009` | FR | 5年收入增长加权排名 | 历史排名越高越高 | 5—20 日 | `rejected` |
| `FR-010` | FR | 异常应计代理 | 现金应计越低越高 | 5—20 日 | `rejected` |
| `FR-011` | FR | 总资产市值比 | 资产/市值越高越高 | 5—20 日 | `rejected` |
| `PV-013` | PV | 12至1个月动量 | 累计收益越高越高 | 1—20 日 | `combination_tested` |
| `PV-014` | PV | 21日CAPM残差波动率 | 残差波动越低越高 | 1—20 日 | `combination_tested` |
| `PV-015` | PV | 36月成交量波动 | 成交量波动越低越高 | 1—20 日 | `rejected` |
| `PV-016` | PV | 36月换手率波动 | 换手波动越低越高 | 1—20 日 | `rejected` |
| `PV-017` | PV | 60月成交量趋势 | 成交量上升趋势越弱越高 | 1—20 日 | `rejected` |
| `PV-018` | PV | 36至13个月长期反转 | 长期收益越低越高 | 1—20 日 | `rejected` |
| `PV-019` | PV | CAPM残差动量代理 | 残差动量越高越高 | 1—20 日 | `combination_tested` |
| `INT-001` | composite `[FR, HF]` | `FR-002/HF-001` 等权截面秩 | 两组件越高越高 | 1 日 | `submitted_smoke` |

表中状态来自本地核验快照上的统一评价：2019—2021 开发，2022、2023
作为两个等权的跨市场状态验证年。长窗口候选允许2019年自然预热，从实际有效日期
开始评价。

## 基础候选定义

### PV

#### PV-001

- 源码：`src/bigalpha2026/candidates/pv/pv_001.py`
- 数据：日频价格、成交额、成交量、成交笔数。
- 机制：相对自身历史，成交活动放大但价格推进有限代表拥挤或供给吸收。
- 主要重复风险：成交活跃度、短期反转、波动率和流动性。

#### PV-002

- 源码：`src/bigalpha2026/candidates/pv/pv_002.py`
- 数据：前收盘、开盘、收盘。
- 机制：衡量隔夜跳空在日内被反向运动吸收的程度。
- 主要失败场景：除权、停复牌、涨跌停和重大新闻跳空。

### HF

#### HF-001

- 源码：`src/bigalpha2026/candidates/hf/hf_001.py`
- 数据：分钟价格、成交额、成交量和成交笔数。
- 机制：识别高活动价格冲击，观察随后 5 分钟的恢复。
- 约束：上午、下午分别计算收益，不跨午休连接。

#### HF-002

- 源码：`src/bigalpha2026/candidates/hf/hf_002.py`
- 数据：分钟价格、成交额、成交量和成交笔数。
- 机制：在成交碎片化条件下衡量价格路径效率。
- 主要重复风险：平均成交规模、零售交易代理和流动性。

### OB

#### OB-001

- 源码：`src/bigalpha2026/candidates/ob/ob_001.py`
- 数据：分钟五档价格和数量。
- 机制：结合有效价差、深度完整性、买卖盘失衡和冲击后深度恢复。
- 约束：价格和数量均为正才算有效档；不能把快照变化解释为撤单。

#### OB-002

- 源码：`src/bigalpha2026/candidates/ob/ob_002.py`
- 数据：分钟五档价格和数量。
- 机制：衡量买卖两侧近端深度形状及尾盘方向持续性。
- 主要重复风险：订单簿不平衡、盘口斜率和 `OB-001`。

### FR

#### FR-001

- 源码：`src/bigalpha2026/candidates/fr/fr_001.py`
- 数据：披露时可见的 TTM 现金流、利润和 LF 总资产。
- 机制：经营现金流相对利润和资产规模的改善。
- 约束：披露日后的首个中国交易日生效，不使用报告期结束日对齐。

#### FR-002

- 源码：`src/bigalpha2026/candidates/fr/fr_002.py`
- 数据：TTM 营业收入、TTM 净利润和 LF 总资产。
- 机制：资产周转和利润资产效率的披露间改善。
- 主要重复风险：ROA、资产周转、成长和公开财务质量因子。

### OAP A/B级候选

这一批的文献来源、字段映射和排除规则见
`reports/oap_signal_screening.md`。现行流程对全部技术有效候选复用统一 S 结果，再相对冻结
screened15 重算 I，并按 S/I 路由进入三条组合管线。`PV-007` 因 21 日离散计数
无法达到每日 50 个不同值，仍为 `technical_reject`。

2026-07-26 按现行规则完成本地真实快照评价。S 要求开发期以及
2022、2023 两个验证年全部通过；I 只使用 2019—2021 开发期，以冻结
screened15、60 日训练/20 日 OOS 评价增量，不使用任一验证年决定成员，也不重复
all36。当前全部获准路由如下：

| 候选 | S | I | 进入管线 |
|---|---:|---:|---|
| `PV-009`、`PV-014` | 通过 | 通过 | 三条全部进入 |
| `PV-003` | 通过 | 未通过 | 仅规则复合 |
| `FR-002`、`FR-004—006`、`OB-001`、`PV-008`、`PV-011`、`PV-013`、`PV-019` | 未通过 | 通过 | Elastic Net、LightGBM |
| 其余技术有效候选 | 未通过 | 未通过 | 淘汰 |
| `PV-007`、`FR-007`、`PV-012` | 技术淘汰 | 不运行 | 不进入 |

三条管线的 2022/2023 `raw_full Rank IC` 分别为：

| 管线 | 2022 | 2023 | 结论 |
|---|---:|---:|---|
| `joint_elastic_net` | 0.06913 | 0.08118 | 当前第一 |
| `self_factor_composite` | 0.05580 | 0.05794 | 第二 |
| `joint_lightgbm` | 0.03067 | 0.02700 | 第三 |

2022 与 2023 的验证地位相同。排序先比较两个验证年中较低的 Rank IC，再比较
两年均值。以上结果未读取 2024，2024 仅在方案冻结后测试。

Elastic Net 的 I 评价与最终训练均对特征和标签做逐日截面标准化，并保存每个
滚动窗权重。

#### PV-003

- 源码：`src/bigalpha2026/candidates/pv/pv_003.py`
- 数据：日频收盘价和前收盘价。
- 机制：过去 21 个交易日最大单日收益越极端，后续收益预期越低。
- 固定方向：按 OAP `MaxRet` 方向取负。

#### PV-004

- 源码：`src/bigalpha2026/candidates/pv/pv_004.py`
- 数据：日频收盘价和前收盘价。
- 机制：过去 21 个交易日日收益的正偏度越高，后续收益预期越低。
- 固定方向：按 OAP `ReturnSkew` 方向取负。

#### PV-005

- 源码：`src/bigalpha2026/candidates/pv/pv_005.py`
- 数据：日收益和成交额。
- 机制：以过去 21 个交易日 `abs(return)/amount` 均值衡量单位成交额价格冲击。
- 主要重复风险：公开波动率、成交额和现有 PV-001 活动强度。

#### PV-006

- 源码：`src/bigalpha2026/candidates/pv/pv_006.py`
- 数据：日频最高价和最低价。
- 机制：Corwin-Schultz 两日高低价价差估计的 21 日均值。
- 约束：这是日线价差估计，不解释为真实盘口报价。

#### PV-007

- 源码：`src/bigalpha2026/candidates/pv/pv_007.py`
- 数据：股票池日频成交量和成交笔数。
- 机制：过去 21 个股票池交易日中零成交状态的比例。
- 约束：缺失记录不自动解释为零成交；停牌影响必须在正式评价中单独报告。

#### FR-003

- 源码：`src/bigalpha2026/candidates/fr/fr_003.py`
- 数据：PIT `lf total_assets`。
- 机制：当前报告期总资产相对去年同报告期增长，按 OAP `AssetGrowth` 取负。
- 约束：仅使用当前披露时已经可见的去年同期记录。

#### FR-004

- 源码：`src/bigalpha2026/candidates/fr/fr_004.py`
- 数据：PIT `ttm operating_revenue`。
- 机制：TTM 营业收入相对去年同报告期的增长。
- 约束：缺少季度每股收入和原论文历史标准化字段，因此只能称为
  OAP-inspired A股改写版，不能称为精确复现。

#### FR-005

- 源码：`src/bigalpha2026/candidates/fr/fr_005.py`
- 数据：PIT `ttm net_cffoa` 和当日 `float_market_cap`。
- 机制：经营现金流相对流通市值越高，预期收益越高。
- 主要重复风险：PE、PB、盈利质量、规模和 FR-001。

## 组合层

### INT-001

- 数据来源：`FR-002 + HF-001`。
- 定义：两个组件分别做日度截面秩，固定 `50%/50%` 合成后再次做截面秩。
- 比赛状态：2026-07-26 公榜分数 `0.57416`，当时团队排名第 `79`；
  私榜尚未公布。
- 定位：端到端冒烟提交，不代表已经通过正式因子准入。
- 冻结记录：`artifacts/frozen/int_001.json`。

## 新候选登记模板

```markdown
### XX-003

- candidate_id:
- data_family: PV / HF / OB / FR / composite
- data_sources:
- mechanism:
- hypothesis:
- source_fields:
- available_time:
- expected_direction:
- expected_horizon:
- failure_conditions:
- duplication_risk:
- status: registered
```
