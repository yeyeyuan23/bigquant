# 因子候选登记表

本文件只保存候选的当前定义、方向、状态和关键结论。数据字段口径见
`data_contract.md`，评价流程与门槛见 `factor_research_plan.md`。

## 登记规则

- 基础因子只有四类：`PV、HF、OB、FR`。
- 跨类组合不是第五类，统一登记为 `composite`，并显式列出 `data_sources`。
- 先登记、后写代码；看过结果后不得在原编号上翻转方向或改变机制。
- `data_checked` 只表示数据可实现，不表示因子有效。
- 所有最终因子均输出 `date、instrument、factor`；提交方向由冻结期对 `z` 与
  `-z` 的 J 比较决定，不能仅因传统收益解释而覆盖高分方向。

实现状态流转：

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
评价完成后，当前组合路由直接写成 `admitted (S)`、`admitted (I)`、
`admitted (T)` 或其组合；字母分别表示进入规则复合、Elastic Net 和
LightGBM。唯一可执行路由以 `reports/factor_pool_admission.csv` 为准。

> 2026-07-27 已将 S/I/T 准入合同升级为 all36 基础代理的 A/B/J 路由增量；
> 最终路线另做一次兄弟路线联合拥挤评分，该场景不是平台全局历史的替代品。
> 下表和现有 `reports/` 仍是升级前的 Rank IC 合同历史结果；在
> `FACTORLIB_ALL36` 快照导出、manifest 核验和完整 J 重评完成前，不得把这些
> 状态当作新合同下的当前路由，也不得据此改写冻结池。

## 当前候选

| ID | 类别 | 机制 | 方向 | 周期 | 当前状态 |
|---|---|---|---|---|---|
| `PV-001` | PV | 活动强度相对价格推进效率 | 推进越有效越高 | 1—3 日 | `admitted (T)` |
| `PV-002` | PV | 隔夜冲击的日内吸收 | 负跳空回补为正，正跳空回落为负 | 1—3 日 | `rejected` |
| `HF-001` | HF | 分钟价格冲击后的吸收与恢复 | 恢复越充分越高 | 1—3 日 | `admitted (S)` |
| `HF-002` | HF | 成交碎片化条件下的价格效率 | 价格效率越高越高 | 1—3 日 | `rejected` |
| `OB-001` | OB | 有效档位盘口韧性 | 价差更低、深度恢复更强越高 | 1 日 | `admitted (T)` |
| `OB-002` | OB | 盘口深度形状的持续偏斜 | 买方近端深度占优越高 | 1 日 | `rejected` |
| `FR-001` | FR | 新披露现金转化质量改善 | 现金转化改善越强越高 | 5—20 日 | `rejected` |
| `FR-002` | FR | 新披露资产效率改善 | 资产周转与 ROA 改善越强越高 | 5—20 日 | `admitted (S/I/T)` |
| `PV-003` | PV | OAP 过去月度最大单日收益 | 最大单日收益越低越高 | 1—20 日 | `admitted (S)` |
| `PV-004` | PV | OAP 日收益偏度 | 偏度越低越高 | 1—20 日 | `rejected` |
| `PV-005` | PV | OAP Amihud 非流动性 | 单位成交额冲击越大越高 | 1—20 日 | `rejected` |
| `PV-006` | PV | OAP Corwin-Schultz 价差估计 | 估计价差越宽越高 | 1—20 日 | `rejected` |
| `PV-007` | PV | OAP 零成交状态 | 零成交日占比越高越高 | 1—20 日 | `technical_reject` |
| `FR-003` | FR | OAP 总资产增长 | 总资产同比增长越低越高 | 5—20 日 | `rejected` |
| `FR-004` | FR | OAP-inspired TTM 收入增长惊喜 | 收入同比增长越高越高 | 5—20 日 | `admitted (S)` |
| `FR-005` | FR | OAP 经营现金流市值比 | 经营现金流/流通市值越高越高 | 5—20 日 | `admitted (S/T)` |
| `FR-006` | FR | OAP-inspired TTM 净利润增长惊喜 | 净利润同比改善越高越高 | 5—20 日 | `admitted (S)` |
| `FR-007` | FR | OAP-inspired 连续盈利改善次数 | 连续改善次数越多越高 | 5—20 日 | `technical_reject` |
| `PV-008` | PV | 52周高点接近度 | 越接近过去52周高点越高 | 1—20 日 | `rejected` |
| `PV-009` | PV | 当期与滞后市场反应延迟 | 价格延迟越低越高 | 1—20 日 | `admitted (S/T)` |
| `PV-010` | PV | 市场协偏度 | 协偏度越低越高 | 1—20 日 | `admitted (S)` |
| `PV-011` | PV | 12至6个月中期动量 | 中期累计收益越高越高 | 1—20 日 | `admitted (S)` |
| `PV-012` | PV | 一级行业动量 | 所属行业过去收益越高越高 | 1—20 日 | `technical_reject` |
| `FR-008` | FR | 48个月盈利增长一致性 | 平均盈利增长越高越高 | 5—20 日 | `rejected` |
| `FR-009` | FR | 5年收入增长加权排名 | 历史排名越高越高 | 5—20 日 | `rejected` |
| `FR-010` | FR | 异常应计代理 | 现金应计越低越高 | 5—20 日 | `rejected` |
| `FR-011` | FR | 总资产市值比 | 资产/市值越高越高 | 5—20 日 | `rejected` |
| `PV-013` | PV | 12至1个月动量 | 累计收益越高越高 | 1—20 日 | `rejected` |
| `PV-014` | PV | 21日CAPM残差波动率 | 残差波动越低越高 | 1—20 日 | `admitted (S/I/T)` |
| `PV-015` | PV | 36月成交量波动 | 成交量波动越低越高 | 1—20 日 | `rejected` |
| `PV-016` | PV | 36月换手率波动 | 换手波动越低越高 | 1—20 日 | `rejected` |
| `PV-017` | PV | 60月成交量趋势 | 成交量上升趋势越弱越高 | 1—20 日 | `rejected` |
| `PV-018` | PV | 36至13个月长期反转 | 长期收益越低越高 | 1—20 日 | `rejected` |
| `PV-019` | PV | CAPM残差动量代理 | 残差动量越高越高 | 1—20 日 | `rejected` |
| `PV-020` | PV | 流动性稀缺条件下的短期反转 | 低成交额状态下的标准化价格冲击越负，因子越高 | 1—3 日 | `admitted (T)` |
| `FR-012` | FR | 营收确认的盈利惊喜 | 盈利与营收惊喜同向且越强，因子绝对值越大 | 5—20 日 | `rejected` |
| `OB-003` | OB | 方向性盘口韧性不对称 | 跌价后买盘恢复相对涨价后卖盘恢复越强越高 | 1—3 日 | `admitted (T)` |
| `FR-013` | FR | 财报披露时点惊喜 | 相对自身同季度历史越早披露越高 | 5—20 日 | `rejected` |
| `PV-021` | PV | 成交量—收益状态切换 | 历史量价状态支持的延续或反转方向越强越高 | 1—3 日 | `rejected` |
| `INT-002` | composite `[FR, PV]` | 财报日异常隔夜反应漂移 | 财报生效日异常隔夜收益越高越高 | 1—20 日 | `rejected` |
| `HF-003` | HF | 分钟相对有符号跳跃 | 下行分钟变差相对占比越高，因子越高 | 1—5 日 | `admitted (S/T)` |
| `HF-004` | HF | 尾盘残余方向成交压力 | 未被同期价格解释的尾盘买压越高，因子越高 | 1 日 | `admitted (T)` |
| `OB-004` | OB | 尾盘盘口失衡创新 | 尾盘买方深度相对全日常态增强越多，因子越高 | 1 日 | `rejected` |
| `PV-022` | PV | 连续信息动量 | 过去12至1个月收益越连续且越高，因子越高 | 1—20 日 | `rejected` |
| `PV-023` | PV | 隔夜上涨—日内回落异常共现 | 两种状态超出独立概率的共现越多，因子越高 | 1—20 日 | `rejected` |
| `FR-014` | FR | PIT 盈利收益率 | 最新可见TTM净利润/当日流通市值越高，因子越高 | 5—20 日 | `admitted (S)` |
| `FR-015` | FR | PIT 净利率同比改善 | 最新TTM净利率相对同季度去年改善越多，因子越高 | 5—20 日 | `admitted (S/T)` |
| `OB-005` | OB | 尾盘微价格压力持续性 | 尾盘微价格偏向买方且方向越持续，因子越高 | 1 日 | `rejected` |
| `INT-003` | composite `[FR, OB]` | 盈利惊喜×事件日流动性摩擦 | 盈利惊喜在尾盘流动性恶化时被增强 | 1—20 日 | `rejected` |
| `INT-001` | composite `[FR, HF]` | `FR-002/HF-001` 等权截面秩 | 两组件越高越高 | 1 日 | `submitted_smoke` |

表中状态来自 `literature_round4_v1_2026-07-27` 本地核验快照上的全量重跑：
2019—2021 决定 S/I/T 准入，2022、2023 只评价冻结组合。长窗口候选允许
2019 年自然预热，从实际有效日期开始评价。

旧合同的 `frozen_I` 为 `FR-002、PV-014`；旧合同的 `frozen_T` 为
`FR-002/005/015、HF-003/004、OB-001/003、PV-001/009/014/020`。
这些成员必须在 J 合同下重新评价，不能自动迁移。两个旧冻结池相互独立；历史
Git 证据为 `reports/factor_pool_decisions.json`，旧本地运行状态另存于
`data/cache/incremental_v3/frozen/frozen_state.json` 和
`data/cache/tree_v3/frozen/frozen_state.json`。新合同使用独立
`single_factor_v1_J、incremental_v4_J、tree_v4_J` 缓存目录，缓存不进入 Git。

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

#### HF-003

- 源码：`src/bigalpha2026/candidates/hf/hf_003.py`
- 数据：日内实现波动率和下行实现波动率。
- 机制：先计算
  `RSJ = 1 - 2 × downside_realized_variance / realized_variance`，再取最近
  5日均值的负值；上涨跳跃相对占优的股票因子更低。
- 固定方向：`-mean_5d(RSJ)`，不根据评价结果翻转。
- 主要重复风险：极端收益、`PV-003`和公开短期波动率。
- status：S、T通过，进入规则复合与 `frozen_T`；I候选级通过但未完成前向和
  池级确认，不进入 Elastic Net。

#### HF-004

- 源码：`src/bigalpha2026/candidates/hf/hf_004.py`
- 数据：尾盘60分钟收益、成交量以及用分钟收益连续分配方向后的成交量。
- 机制：先以`尾盘方向成交量/尾盘成交量`得到方向压力，再用同一股票严格滞后的
  60日、至少30日回归剔除同期尾盘收益能够解释的部分。
- 固定方向：当前尾盘方向压力减去历史价格—压力关系的拟合值，残余买压越大越高。
- 约束：方向成交量是固定 logistic BVC 近似，不是真实主动买卖标记；新增聚合字段
  必须先在AIStudio短窗核验再生成全量面板。
- 主要重复风险：公开`netflow_amount_rate_main`和普通成交量不平衡。
- status：S、I未通过；T通过并进入 `frozen_T`。

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

#### OB-003

- 源码：`src/bigalpha2026/candidates/ob/ob_003.py`
- 数据：负向中间价冲击后的买盘深度恢复、正向中间价冲击后的卖盘深度恢复。
- 机制：比较两种方向性流动性供给；买盘补充相对更强代表下行冲击更易被吸收。
- 固定方向：每日截面 `rank(negative-shock bid recovery) - rank(positive-shock ask recovery)`。
- 主要重复风险：订单簿恢复、买卖盘不平衡和 `OB-001`。

#### OB-004

- 源码：`src/bigalpha2026/candidates/ob/ob_004.py`
- 数据：全日盘口深度失衡中位数、日内失衡标准差和尾盘60分钟失衡中位数。
- 机制：
  `(tail_imbalance - full_day_imbalance) / full_day_imbalance_std`，只衡量
  尾盘盘口状态相对当日常态的创新。
- 固定方向：尾盘买方深度相对增强越多，因子越高。
- 约束：分钟盘口是快照，不能把状态变化解释为新增委托或撤单。
- 主要重复风险：`OB-001`中的尾盘失衡水平和`OB-002`的盘口形状。
- status：S/I/T均未通过，不进入组合。

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

#### FR-012

- 源码：`src/bigalpha2026/candidates/fr/fr_012.py`
- 数据：披露时可见的 TTM 净利润和 TTM 营业收入。
- 机制：分别用同一股票过去 8 次、至少 4 次已披露同比变化估计惊喜；只有盈利与
  营收惊喜同向时保留，强度为两者绝对值乘积的平方根。
- 固定方向：同向为正时因子为正，同向为负时因子为负，方向冲突时为 0。
- 约束：历史标准化窗口整体滞后一期，按 `effective_date` 向后 as-of。

### 文献机制候选

#### PV-020

- 源码：`src/bigalpha2026/candidates/pv/pv_020.py`
- 数据：收盘价、前收盘价和成交额。
- 机制：用前 20 日波动率标准化当日收益，再用“前 20 日成交额中位数/当日成交额”
  放大流动性稀缺状态下的反转。
- 固定方向：`-return_shock * liquidity_scarcity`，两个历史尺度均整体滞后一期。
- 主要重复风险：普通短期反转、Amihud 非流动性和 `PV-005`。
- status：S未通过；I候选级通过但未完成前向和池级确认；T通过并进入
  `frozen_T`。

### OAP A/B级候选

这一批的文献来源、字段映射和排除规则见
`reports/oap_signal_screening.md`。现行流程对全部技术有效候选复用统一 S 结果，再相对冻结
screened15 重算 I。S 只控制规则复合，I 只控制 Elastic Net；LightGBM 使用独立
树增量 T。`PV-007` 因 21 日离散计数
无法达到每日 50 个不同值，仍为 `technical_reject`。

现行正式口径为：S 只使用 2019—2021 总体表现和月度稳定性；I 与 T 使用同一
60日训练、20日样本外窗口，并同样只在开发期冻结准入。2022、2023 只报告，不
反向决定成员。通用微观数据底座更新后，旧路由和旧组合分数全部失效；新的获准
列表与结果只能由全量重跑后的报告写回本节。

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

### 已提交版本与当前本地候选

| pipeline / version | 冻结输入 | 2022 Rank IC | 2023 Rank IC | Notebook | 比赛状态 |
| --- | --- | ---: | ---: | --- | --- |
| `self_factor_composite / submitted-v1` | `FR-002/004/005/006 + PV-003/009/011/014`，家族内等权后 FR/PV 等权 | 0.04087 | 0.03650 | `submissions/rule_v02.ipynb` | 2026-07-26 已提交，保留为历史对照 |
| `self_factor_composite / local-current` | `FR-002/004/005/006/014/015 + HF-001/003 + PV-003/009/010/011/014`，家族内等权后 FR/HF/PV 等权 | 0.05464 | 0.05178 | 尚未生成新提交 Notebook | 新 S 规则本地重跑通过，尚未提交 |
| `joint_elastic_net / local-current` | 冻结 `screened15 + FR-002 + PV-014`；日度截面秩目标、非负系数 | 0.07334 | 0.07020 | `submissions/enet_v01.ipynb` | 新 I 合同完整重评并冻结 |
| `joint_lightgbm / local-current` | 冻结 `screened15` 加 `FR-002/005/015, HF-003/004, OB-001/003, PV-001/009/014/020`；日度截面秩目标、正单调约束 | 0.07377 | 0.06965 | `submissions/lgbm_v01.ipynb` | 新 T 合同完整重评并冻结 |

已提交 Notebook 均仅保留比赛要求的
`main(datasources, start_date, end_date)`，返回列固定为
`date, instrument, factor`。`local-current` 只代表本地冻结评价结果，不表示已经
通过 AIStudio 或比赛提交验收。

### INT-001

- 数据来源：`FR-002 + HF-001`。
- 定义：两个组件分别做日度截面秩，固定 `50%/50%` 合成后再次做截面秩。
- 比赛状态：2026-07-26 公榜分数 `0.57416`，当时团队排名第 `79`；
  私榜尚未公布。
- 定位：端到端冒烟提交，不代表已经通过正式因子准入。
- 冻结记录：`artifacts/frozen/int_001.json`。

### FR-013

- candidate_id：`FR-013`
- data_family：`FR`
- data_sources：`bigalpha_2026_financial`
- mechanism：公司会相对自身正常节奏提前披露好消息、推迟披露坏消息。
- source_fields：`disclosure_date、effective_date、report_date、category、shift`
- formula：当前披露滞后天数减去同一股票、同一报告季度过去 3 次披露滞后中位数，
  再取负；历史不足 2 次时不产生信号。
- available_time：`effective_date`
- expected_direction：异常提前披露为正，异常延迟披露为负。
- expected_horizon：5—20 日。
- failure_conditions：法定截止日聚集压过公司自主时点、报告期类型映射不稳定。
- duplication_risk：低；现有 FR 候选均以财务数值而非披露行为为核心。
- status：S/I/T均未通过，不进入组合。

### PV-021

- candidate_id：`PV-021`
- data_family：`PV`
- data_sources：`bigalpha_2026_stock_bar1m` 日级聚合
- mechanism：高成交状态下的收益可能来自信息交易并延续，低成交状态更可能是
  流动性冲击并反转；每只股票用自身历史估计状态方向。
- source_fields：`close、pre_close、amount`
- formula：先用滞后 20 日成交额中位数构造异常成交额；再用截至前一日的 120 日、
  至少 60 日窗口估计
  `r_t = a + b*r_(t-1) + c*r_(t-1)*abnormal_volume_t`，当日信号为
  `c_hat_(t-1)*r_t*abnormal_volume_t`。
- available_time：当日收盘后。
- expected_direction：`c_hat` 为正时顺势，为负时反转。
- expected_horizon：1—3 日。
- failure_conditions：滚动回归病态、异常成交额由停复牌或公司行动驱动。
- duplication_risk：中低；与普通动量/反转共享收益输入，但方向由历史量价状态决定。
- status：S/I/T均未通过，不进入组合。

### INT-002

- candidate_id：`INT-002`
- data_family：`composite`
- data_sources：`[FR, PV]`
- mechanism：财报披露后的异常隔夜反应包含价值信息，并在后续交易日继续漂移。
- source_fields：`effective_date、instrument、open、pre_close`
- formula：在财报 `effective_date` 计算个股隔夜收益减当日股票池等权隔夜收益，
  固定保留 20 个交易日；不叠加盈利方向过滤。
- available_time：财报生效日开盘后；作为日因子在当日收盘后使用。
- expected_direction：异常隔夜收益越高，因子越高。
- expected_horizon：1—20 日。
- failure_conditions：开盘价缺失、停牌复牌跳空、市场隔夜基准受极端横截面污染。
- duplication_risk：中；与 `PV-002` 共享隔夜收益，但只在财报事件窗口激活。
- status：S 通过，但复合候选不递归进入规则复合；I/T未通过，最终不进入三条
  组合管线。

### PV-022

- candidate_id：`PV-022`
- data_family：`PV`
- data_sources：`bigalpha_2026_stock_bar1m` 日级聚合
- mechanism：把过去12至1个月动量按日收益符号的连续程度加权，区分持续小幅
  信息进入和少数离散跳跃造成的相同累计收益。
- source_fields：`close、pre_close`
- formula：先算 `PRET = close_(t-21) / close_(t-252) - 1`，再算
  `ID = sign(PRET) × (negative_share - positive_share)`，最终为
  `PRET × (1-ID)/2`。
- available_time：当日收盘后，形成期严格截止到21个交易日前。
- expected_direction：连续形成的正动量越强越高；连续负动量为负。
- expected_horizon：1—20 日。
- failure_conditions：长期停牌导致形成窗稀疏、除权价格未正确复权。
- duplication_risk：中；与普通12—1月动量共享累计收益，但新增路径连续性。
- status：S/I/T均未通过，不进入组合。

### PV-023

- candidate_id：`PV-023`
- data_family：`PV`
- data_sources：`bigalpha_2026_stock_bar1m` 日级聚合
- mechanism：识别正隔夜收益与负日内收益之间超出各自边际频率的异常共现，
  表达隔夜与日内投资者的方向分歧。
- source_fields：`open、close、pre_close`
- formula：20日滚动
  `P(overnight>0, intraday<0) - P(overnight>0)×P(intraday<0)`。
- available_time：当日收盘后。
- expected_direction：高开后回落的异常共现越多，因子越高。
- expected_horizon：1—20 日。
- failure_conditions：涨跌停、停复牌和公司行动扭曲开盘收益。
- duplication_risk：中低；与 `PV-002` 共享隔夜和日内收益，但使用滚动共现结构。
- status：S/I/T均未通过，不进入组合。

### FR-014

- candidate_id：`FR-014`
- data_family：`FR`
- data_sources：`bigalpha_2026_financial + daily exposures`
- mechanism：以披露时可见的TTM盈利相对同日可交易流通市值衡量便宜程度。
- source_fields：`net_profit、effective_date、float_market_cap`
- formula：`PIT TTM net_profit / same-day float_market_cap`。
- available_time：财报 `effective_date` 当日收盘后。
- expected_direction：盈利收益率越高，因子越高。
- expected_horizon：5—20 日。
- failure_conditions：负利润、极小流通市值和财报/市值单位不一致。
- duplication_risk：中；接近价值因子，但采用严格PIT盈利与流通市值。
- status：S通过，进入规则复合；I/T未通过。

### FR-015

- candidate_id：`FR-015`
- data_family：`FR`
- data_sources：`bigalpha_2026_financial`
- mechanism：净利率的同季度同比改善同时约束盈利和营收，减少单纯利润增长的
  规模效应。
- source_fields：`net_profit、operating_revenue、report_date、effective_date`
- formula：最新 `TTM net_profit / TTM operating_revenue` 减去同一报告季度
  上一年的净利率。
- available_time：财报 `effective_date`。
- expected_direction：净利率同比改善越多，因子越高。
- expected_horizon：5—20 日。
- failure_conditions：收入接近零、同季度历史缺失、主营业务发生结构性变化。
- duplication_risk：中；与盈利增长和资产效率共享基本面信息，但直接刻画利润率。
- status：S、T通过，进入规则复合与 `frozen_T`；I候选级通过但未完成前向和
  池级确认，不进入 Elastic Net。

### OB-005

- candidate_id：`OB-005`
- data_family：`OB`
- data_sources：`bigalpha_2026_stock_bar1m`
- mechanism：一档微价格相对中间价的位置反映两侧最优深度压力；尾盘方向持续
  出现时，比单点盘口失衡更可靠。
- source_fields：`bid_price1、ask_price1、bid_volume1、ask_volume1`
- formula：分钟
  `gap=(microprice-mid)/(ask1-bid1)`，日级信号为
  `tail60_median(gap) × abs(tail60_mean(sign(gap)))`。
- available_time：当日收盘后。
- expected_direction：尾盘持续买方压力为正，持续卖方压力为负。
- expected_horizon：1 日。
- failure_conditions：一档报价无效、价差为零、尾盘有效快照不足。
- duplication_risk：中；与盘口失衡共享深度输入，但微价格按对侧价格加权。
- status：S/I/T均未通过，不进入组合。

### INT-003

- candidate_id：`INT-003`
- data_family：`composite`
- data_sources：`[FR, OB]`
- mechanism：事件日尾盘价差扩大且深度收缩时，盈利惊喜更可能尚未被充分吸收。
- source_fields：`FR-012 factor_raw、full/tail spread、full/tail depth`
- formula：
  `FR-012 × (1 + positive_rank(log(tail_spread/full_spread) + log(full_depth/tail_depth)))`，
  固定保留20个交易日。
- available_time：财报生效日收盘后。
- expected_direction：正盈利惊喜被流动性恶化增强，负惊喜同方向放大。
- expected_horizon：1—20 日。
- failure_conditions：事件日无盘口快照、价差或深度无效、财报惊喜历史不足。
- duplication_risk：中低；只在财报事件上将基本面与当日流动性状态交互。
- status：S/I未通过；开发期有效日239，低于T门槛240，不进入组合。

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
