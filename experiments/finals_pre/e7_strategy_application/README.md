# E7：冻结因子在不同持仓规则下的应用比较

## 2026-09-16 演示稿使用的结果

当前正文展示每日五分位、排名缓冲区和五日平均排名，分别统计 2025 年全年、2026 年至 8 月 28 日的累计收益。五日规则在已查看时期中选出，现有结果属于回顾性比较。

- [候选规则、成本情景和逐日账本](../../../reports/dependencies/finals_pre/e7_strategy_application/20260916_simple_rules/README.md)
- [按年末收盘切分的扣费后收益与换手](../../../reports/dependencies/finals_pre/e7_strategy_application/20260916_calendar_year_returns/README.md)
- [相同规则下的不扣费收益](../../../reports/dependencies/finals_pre/e7_strategy_application/20260916_calendar_year_gross/README.md)
- [五日排名窗口与调仓时点检查](../../../reports/dependencies/finals_pre/e7_strategy_application/20260916_rank_mean_timing/README.md)

本地回测输入保存在 `data/runtime/finals_pre/e7_strategy_application/20260916/inputs/`，不提交原始 parquet。`compare_simple_strategies.py` 生成候选比较，`yearly_close_returns.py` 生成分年结果，`audit_rank_mean_timing.py` 检查历史信息使用。对应 HTML 制作脚本在 PRE 仓库读取上述结果。

## 研究问题与配置

固定因子、股票池、评价期与费率，比较降低换手能否抵偿跟踪最新信号变慢造成的毛收益变化。四种规则全部报告，不按回测表现选择费率或调仓起点。

| 规则 | 定义 |
|---|---|
| 每日五分位 | 前 20% 等权做多，后 20% 等权做空，每日调整 |
| 排名缓冲区 | 原多头仍在前 30%、原空头仍在后 30% 时优先保留，按排名填满各 20%；仍按日等权调整 |
| 每三日调仓 | 首日建仓，此后每三日更新；非调仓日持有原股数 |
| 每日调整 50% | 当前实际权重与最新五分位目标各取一半，再将多空两侧分别归一化；不表示成交量恰好减半 |

多头和空头目标分别为扣费后净值的 +100%、−100%。三日规则主起点固定为第一个可交易日，另完整报告后移 1、2 日的相位；各相位均在首日建仓。

## 输入与持有区间

使用正式提交的冻结因子和原评价中性化方法：当天 10 项 Barra 风格与 32 项行业暴露。权重训练区间为 2019-01-02～2024-12-26，checkpoint SHA-256 为 `252c39baf946f494898fcd0e2c3a0aeca4de2ece378b9de6386a5200378e1dcb`。

- 原始冻结因子共 402 日、402,000 行；原 O2C 评价复现 401 日 RankIC `0.02516923999851031`。
- 当天收盘获得分数，次日开盘调整，此后持有到下一开盘；所有隔夜涨跌均计入。
- 应用期为 **2025-01-03 开盘至 2026-08-28 开盘，共 400 个持有区间**；最后一个可用开盘用于可以成交部分的期末平仓。
- 开盘参考价取 09:31 分钟 bar 的 open × adjust_factor，复权开盘到开盘收益与旧标签逐键核对最大误差为 0。
- 模型预测目标仍为 O2C；本应用实验含隔夜，不替代 401 日 O2C 排序评价。

## 费用与滑点：本次修订

基准采用 **买入 3bp、卖出 8bp**：买卖佣金各 3bp，卖出另计 5bp 印花税。该比例参照 [BigTrader 官方文档示例](https://fund.bigquant.com/wiki/doc/3gG2rg4jBd)，卖出印花税参照 [上交所说明](https://one.sse.com.cn/onething/gptz/)；核对日期为 2026-09-06。这是有来源的比例费率假设，不是统一行业标准或实际账户费率校准。

| 情景 | 买入费率 | 卖出费率 | 每侧额外滑点 | 买入/卖出合计系数 |
|---|---:|---:|---:|---:|
| 毛收益参照 | 0bp | 0bp | 0bp | 0 / 0bp |
| 基准 | 3bp | 8bp | 0bp | 3 / 8bp |
| 滑点 2bp | 3bp | 8bp | 2bp | 5 / 10bp |
| 滑点 5bp | 3bp | 8bp | 5bp | 8 / 13bp |

买卖方向由持仓变化决定：买入增加多头和买回平空均算买入；卖出多头和开空仓均算卖出。建仓、调整及最后可以成交的平仓全部计费。

令 B、S 为开盘参考价下的买入、卖出成交金额，s 为每侧滑点基点数：

`扣费金额 = 0.0003 × B + 0.0008 × S + (s / 10000) × (B + S)`。

滑点以参考成交金额的线性现金损耗表示，不改动行情或信号；费用与滑点分别记账。未计算滑点与税费之间的二阶交叉项。每日目标按扣费后净值求解，避免费用导致额外杠杆或凭空现金。

**尚未计入平台示例中的每笔最低 5 元佣金**：本实验从单位净值出发，没有指定资金规模；也未额外加入过户费。不能把此比例模型称为复现平台全部费用。未模拟借券/融资费、非线性冲击、涨跌停排队或开盘容量，且假设可借到空头股票。复权价格用于近似公司行动，未逐笔重建分红现金与原始股数。

缺开盘报价时不成交，以当时最近可见价估值；不使用未来价格回填，也不根据未来收益是否存在筛选股票。期末无法成交的持仓继续估值并单独报告。

日均单边换手为 `0.5 × (B + S) / 调仓前净值`；不能直接乘单一成本系数代替上述买卖分项费用。年化收益为日收益均值 × 252，另保留 CAGR；Sharpe 使用日收益样本标准差，最大回撤包含初始净值 1。毛收益来自独立的零成本记账。

## 当前结果与历史记录

新费率结果见 [20260906_fee_slippage](../../../reports/dependencies/finals_pre/e7_strategy_application/20260906_fee_slippage/README.md)。四规则 × 四情景，加两种基准费率的三日调仓相位，共 18 组、7,200 行逐日账本。

此前每侧统一 0/5/10/20bp 的结果保留在 `reports/dependencies/finals_pre/e7_strategy_application/20260906/`，原脚本保留在该目录 `code/`。该历史目录不再作为当前正文的数据源。此次修改由用户在查看旧结果后要求，不能声称新费率在所有研究开始前就已预注册；新结果仍是同一已查看时期的应用复盘。策略选择需要后续未参与研究的时期验证。

## 运行与验证

复用 AutoDL 已准备的输入，无需重新训练或整理分钟行情：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 python backtest.py \
  --inputs /root/autodl-tmp/strategy-application-20260906/results \
  --out /root/autodl-tmp/strategy-application-20260906/fee_slippage_results
python -m unittest test_backtest -v
```

从仓库根目录验证已取回结果（不需要原始行情）：

```sh
python experiments/finals_pre/e7_strategy_application/validate_results.py \
  --results reports/dependencies/finals_pre/e7_strategy_application/20260906_fee_slippage
```

12 项测试覆盖可配置调仓间隔、买卖不对称费率、开空/平空方向、滑点分项、初末费用、价格漂移、非调仓日持股、未来信息隔离、缺报价交易阻断、终端未平仓与自融资恒等式。独立验证器逐日重算费用、滑点、净值和汇总指标，并核对传输文件与实际运行脚本哈希。2026-09-16 的复核使用上述本地输入，汇总、逐日净值和审计文件纳入研究记录。
