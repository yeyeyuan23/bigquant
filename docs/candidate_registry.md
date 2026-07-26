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
| `PV-001` | PV | 活动强度相对价格推进效率 | 推进越有效越高 | 1—3 日 | `no_registered_direction_evidence` |
| `PV-002` | PV | 隔夜冲击的日内吸收 | 负跳空回补为正，正跳空回落为负 | 1—3 日 | `no_registered_direction_evidence` |
| `HF-001` | HF | 分钟价格冲击后的吸收与恢复 | 恢复越充分越高 | 1—3 日 | `conditional_watch` |
| `HF-002` | HF | 成交碎片化条件下的价格效率 | 价格效率越高越高 | 1—3 日 | `no_registered_direction_evidence` |
| `OB-001` | OB | 有效档位盘口韧性 | 价差更低、深度恢复更强越高 | 1 日 | `no_registered_direction_evidence` |
| `OB-002` | OB | 盘口深度形状的持续偏斜 | 买方近端深度占优越高 | 1 日 | `no_registered_direction_evidence` |
| `FR-001` | FR | 新披露现金转化质量改善 | 现金转化改善越强越高 | 5—20 日 | `no_registered_direction_evidence` |
| `FR-002` | FR | 新披露资产效率改善 | 资产周转与 ROA 改善越强越高 | 5—20 日 | `development_survivor` |
| `INT-001` | composite `[FR, HF]` | `FR-002/HF-001` 等权截面秩 | 两组件越高越高 | 1 日 | `submitted_smoke` |
| `MICRO-EN-001` | composite `[HF, OB]` | 微观结构相对公开因子库的滚动 Elastic Net 增量 | 由训练窗学习 | 1 日 | `development_watch` |

## 基础候选定义

### PV

#### PV-001

- 源码：`src/bigalpha2026/candidates/pv/pv_001.py`
- 数据：日频价格、成交额、成交量、成交笔数。
- 机制：相对自身历史，成交活动放大但价格推进有限代表拥挤或供给吸收。
- 主要重复风险：换手、短期反转、波动率和流动性。

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

## 组合层

### INT-001

- 数据来源：`FR-002 + HF-001`。
- 定义：两个组件分别做日度截面秩，固定 `50%/50%` 合成后再次做截面秩。
- 比赛状态：2026-07-26 公榜分数 `0.57416`，当时团队排名第 `79`；
  私榜尚未公布。
- 定位：端到端冒烟提交，不代表已经通过成本和严格交易准入。

### MICRO-EN-001

- 数据来源：10 个 HF/OB 日级组件和比赛公开基础因子库。
- 定义：滚动训练“公开库 + 微观结构”与“仅公开库”两个 Elastic Net，
  两者预测差作为增量因子。
- 当前开发结果：140 个 OOS 日中，基础库 Rank IC `0.01052`，增强后
  `0.02471`，增量 `0.01419`；5/7 个滚动窗口改善。
- 限制：当前结果包含已经查看过的 2023 代表月份；训练、推理和独立留出
  合同尚未冻结，因此不能提交。
- 实现位置：尚未创建；冻结后应放入
  `src/bigalpha2026/candidates/composite/`。

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
