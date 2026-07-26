# OAP A/B 候选实现表

本文是 `candidates/` 目录内的实现索引。OAP 只提供机制和原始定义；代码只读取
比赛本地快照，不读取 OAP 股票级信号。所有输出均为
`date、instrument、factor`，且值越大代表预期收益越高。

实现忠实度：

- `exact`：核心字段、方向和窗口可直接重建；
- `near_exact`：保留公式，仅替换为A股市场或交易口径；
- `adapted`：缺少原字段，保留机制并明确改写；
- `proxy`：只能构造机制代理，不称为论文复现。

## A级：15个，已全部实现

| ID | OAP信号 | 类别 | 忠实度 | 本地定义 | 固定方向 | S/I |
|---|---|---|---|---|---|---|
| `FR-003` | AssetGrowth | FR | exact | PIT总资产同比增长 | 取负 | 未通过/未通过 |
| `FR-005` | cfp | FR | near_exact | PIT经营现金流/流通市值 | 正 | 未通过/通过 |
| `FR-006` | EarningsSurprise | FR | adapted | PIT TTM净利润同比变化 | 正 | 未通过/通过 |
| `FR-004` | RevenueSurprise | FR | adapted | PIT TTM营业收入同比变化 | 正 | 未通过/通过 |
| `FR-007` | NumEarnIncrease | FR | adapted | PIT TTM净利润连续改善次数 | 正 | 技术淘汰/不运行 |
| `PV-003` | MaxRet | PV | exact | 21日最大单日收益 | 取负 | 通过/未通过 |
| `PV-004` | ReturnSkew | PV | exact | 21日日收益偏度 | 取负 | 未通过/未通过 |
| `PV-008` | High52 | PV | exact | 收盘价/252日最高收盘价 | 正 | 未通过/通过 |
| `PV-005` | Illiquidity | PV | near_exact | 21日平均绝对收益/成交额 | 正 | 未通过/未通过 |
| `PV-006` | BidAskSpread | PV | exact | Corwin-Schultz日线价差估计 | 正 | 未通过/未通过 |
| `PV-009` | PriceDelayRsq | PV | adapted | 当期及4阶滞后市场相关解释份额 | 延迟取负 | 通过/通过 |
| `PV-010` | CoskewACX | PV | near_exact | 252日个股收益与市场平方收益协偏度 | 取负 | 未通过/未通过 |
| `PV-007` | zerotrade1M | PV | adapted | 21日零成交状态比例 | 正 | 技术淘汰/不运行 |
| `PV-011` | IntMom | PV | exact | `t-252`至`t-126`收益 | 正 | 未通过/通过 |
| `PV-012` | IndMom | PV | near_exact | 一级行业126日市值加权收益 | 正 | 技术淘汰/不运行 |

## B级：11个，已全部实现

| ID | OAP信号 | 类别 | 忠实度 | 本地定义与窗口 | 固定方向 | S/I |
|---|---|---|---|---|---|---|
| `FR-008` | EarningsConsistency | FR | adapted | 16次季度披露、约48个月的TTM盈利增长均值 | 正 | 未通过/未通过 |
| `FR-009` | MeanRankRevGrowth | FR | near_exact | 同报告期安全生效后的5年收入增长加权排名 | 正 | 未通过/未通过 |
| `FR-010` | AbnormalAccruals | FR | proxy | `-(净利润-经营现金流)/总资产`，缺少PPE回归 | 取负 | 未通过/未通过 |
| `FR-011` | AM | FR | exact | PIT总资产/当日流通市值 | 正 | 未通过/未通过 |
| `PV-013` | Mom12m | PV | exact | `t-252`至`t-21`收益 | 正 | 未通过/通过 |
| `PV-014` | RealizedVol | PV | near_exact | 21日CAPM残差波动率 | 取负 | 通过/通过 |
| `PV-015` | VolSD | PV | near_exact | 月成交量36月滚动标准差，至少24月 | 取负 | 未通过/未通过 |
| `PV-016` | std_turn | PV | near_exact | 月换手率36月滚动标准差，至少24月 | 取负 | 未通过/未通过 |
| `PV-017` | VolumeTrend | PV | adapted | 月成交量对时间的60月斜率/均值，至少30月 | 取负 | 未通过/未通过 |
| `PV-018` | LRreversal | PV | exact | `t-756`至`t-273`收益 | 取负 | 未通过/未通过 |
| `PV-019` | ResidualMomentum | PV | proxy | CAPM残差、跳过21日后的231日均值/标准差 | 正 | 未通过/通过 |

## 运行规则

- 长窗口保持原长度；2019年允许作为自然预热期，不补拉2018年价格数据。
- 预热期无值不算技术失败，但没有开发期或没有60日训练窗时，S/I明确失败。
- `S=通过` 进入规则复合；`I=通过` 进入Elastic Net和LightGBM。
- 同时通过S和I时进入三条管线；两者均未通过时淘汰。
- 公式、方向或窗口如需改变，必须登记新编号，不能覆盖已看过结果的编号。
