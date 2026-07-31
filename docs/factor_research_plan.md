# 因子研究与评价流程

本文件只定义仍在使用的研究流程、时间切分和准入门槛。具体候选定义见
`candidate_registry.md`，字段与时点见 `data_contract.md`。

## 1. 总原则

- 先登记机制和方向，再实现和查看结果。
- 本地是因子代码、评价、组合训练、规则和版本的事实来源；AIStudio 是真实数据
  查询、字段核验和最终平台验收的事实来源。
- 基础因子只分 `PV、HF、OB、FR` 四类；跨类候选进入 `composite`。
- 所有模型按日期切分，禁止随机拆分股票日。
- 单因子表现一般但有稳定组合增量时可以保留；没有单因子或组合增量时淘汰。
- 公榜只验证冻结候选，不能作为无约束调参集。

## 2. 环境分工与交互闭环

### 2.1 本地职责

本地仓库负责：

- 登记候选机制、方向、字段、可用时点和失败条件；
- 实现候选、评价、组合和 Notebook 生成代码；
- 运行单元测试、接口测试、防泄漏测试和合成数据 `--check`；
- 在已核验并冻结的数据快照上运行单因子、screened15 增量和三条组合管线；
- 固化时间切分、准入门槛、模型参数和冻结版本；
- 生成标准报告，更新候选状态并决定下一轮研究；
- 使用 Git 分支、MR/PR 和冻结产物管理最终版本。

本地结果只有在输入来自 AIStudio、通过 manifest 和数据合同核验且快照未被修改时，
才可作为正式研究结论；合成数据或未核验数据仍只用于排错和预筛。

### 2.2 AIStudio 职责

AIStudio 负责：

- 查询并核验比赛股票池、分钟行情、财务披露、风险暴露和冻结的 screened15；
- 将分钟、盘口和 PIT 财务数据按通用合同聚合为可复用日级面板；
- 在真实股票池上检查三列输出、覆盖率、重复键、时点和泄漏；
- 首次筛选并冻结 `screened15`，之后只导出这 15 列及其 manifest；
- 对冻结赢家做短窗复跑，生成提交 Notebook 和比赛提交结果。

Notebook 只能执行本地已经版本化的规则，不能临时修改方向、时间切分、门槛或
模型参数。原始分钟表默认留在 AIStudio；经过核验的日级股票池、标签、共享面板、
screened15 和风险暴露可以作为同队研究快照同步到本地，但不得进入 Git。

### 2.3 比赛网页职责

比赛网页只用于上传已冻结且通过短窗验收的 Notebook，以及查看提交状态、分数和
排名。公榜反馈不回流为无约束调参标签。

### 2.4 固定交互循环

```text
本地登记并实现候选
→ 本地测试通过并固定 Git 版本
→ AIStudio 只补齐并核验候选需要的数据快照
→ 快照和 manifest 同步到本地
→ 本地运行单因子、screened15 增量和三条组合管线
→ 本地登记保留、淘汰或下一轮机制
→ 冻结赢家回到 AIStudio 做短窗验收和提交
```

本地传给 AIStudio：

- 候选或组合代码及 Git 版本；
- 候选登记信息；
- 固定的时间切分、门槛和模型配置。

AIStudio 回传本地：

```text
data/ 下经过核验的日级 Parquet
对应 manifest、字段合同和查询版本
```

本地生成的标准报告文件集合见第 10 节。原始分钟表不回传。双方使用同一份
E2E 一分钟数据开发新逻辑时，候选 PR 必须交付最小可复现生成代码、
字段合同和候选到派生列的 manifest，不上传原始或派生数据。

如果新候选需要当前合同外的平台数据，AIStudio 还必须交付可复现取数代码、
更新后的数据合同和 manifest；数据同步方式由负责人确认。只有截图或单个 IC
数字不能形成可复现交付。

## 3. 时间切分

| 用途 | 时间 |
|---|---|
| 技术冒烟 | 2022 短区间，只检查接口、覆盖率和防泄漏 |
| 开发与方法/参数选择 | 2019—2022 |
| 年度 J 评估 | 2023、2024 |
| 官方评价 | 公榜与私榜 |

已经查看的区间不得再次用于修改同一候选的方向或公式。需要研究相反机制时，
使用新编号并在未查看区间验证。

滚动长窗口因子不要求为了补齐开发期第一年而额外下载前史。若当前快照从2019年
开始，则2019年可作为自然预热期，保持原窗口不变，从首次产生有效值的日期开始
计算S和I；预热期缺失不算技术失败。只有扣除预热后样本仍低于统一门槛时才淘汰。
当前已登记候选均使用本地已有快照研究，不再为这些候选补拉2018年价格数据。

HF/OB 首轮可以使用冻结的代表月份以控制计算成本；候选通过内部门槛后再补完整
年份。月份不得根据候选收益表现挑选。

## 4. 候选开发流程

### 4.1 登记

在 `candidate_registry.md` 中写清：

- 编号、数据族和数据来源；
- 经济机制与预期方向；
- 可用时点和预测周期；
- 必要字段；
- 失败条件与重复风险。

### 4.2 实现

- 文件放入 `src/bigalpha2026/candidates/<family>/`。
- 一个候选一个模块，例如 `hf/hf_003.py`。
- 数据读取、特征计算和三列输出分离。
- 在对应 `__init__.py` 导出公开函数。
- 添加同编号测试。

### 4.3 技术门槛

候选必须满足：

- 输出严格为 `date、instrument、factor`；
- 无重复股票日期；
- 原始特征覆盖率原则上不低于 95%；
- 每日有效截面值不少于 50；
- 因子值有限；
- 股票池和交易日完整；
- 未来数据扰动不改变过去因子。

技术通过只代表“可计算”，不代表“有效”。

## 5. 单因子评价

统一入口：

- 准入模块：`src/bigalpha2026/single_factor_admission.py`
- 核心入口：`run_single_factor_admission`
- 指标原语：`src/bigalpha2026/evaluation.py`
- 编排脚本：`scripts/run_first_round.py`

AIStudio 只负责拉取和核验真实数据快照；单因子指标、I 增量和组合训练均在本地
对已核验快照运行。

必须报告：

- 日度 Rank IC 均值、IC_IR、t 值和正向比例；
- 年度或代表月份稳定性；
- 五分组单调性；
- 多空收益和夏普；
- 市值、流动性和可交易子集；
- 行业、规模、流动性中性化后的结果；
- 与同族候选和公开基础因子的 Rank 相关性。

S 的正式准入由 `src/bigalpha2026/single_factor_admission.py` 的
`run_single_factor_route_admission` 编排，只读取 2019—2022 开发期；
2023、2024 只生成验证报告，不得反向改变 S。

S 是规则复合，没有模型自动压低坏因子权重，因此先做严格 trial gate：

- `coverage >= 0.95`；
- 至少 120 个有截面离散度的开发日；
- 开发期日度 Rank IC 均值不低于 `0.01`；
- 年度 fold 最差 Rank IC 不低于 `-0.005`；
- 年度正向比例和方向一致性均不低于 `0.60`；
- 与当前 S baseline 的最大绝对 Rank 相关性不高于 `0.85`。
- 去行业固定效应和可用 BARRA 风格后的 Rank IC 均值不低于 `0.005`；
- 至少覆盖 10 个行业，每个行业至少 40 个有效日，行业内 IC 同向比例不低于
  `0.55`。

满足这些质量、稳定性、中性化有效性和低重复要求后，候选进入最终 S。这里不会
因为因子存在风格暴露就直接否决，而是检查剔除行业和风格暴露后是否仍有效。S 因此是：

```text
单因子质量合格、强且稳定
→ 行业内普遍有效，且中性化后仍有效
→ 与当前 S 不过度重复
→ 冻结进 S，成为 I 的唯一候选来源
```

准入阶段不计算本地 J；可交易、分组和压力期结果继续作为人工诊断。

单因子评价只产生明确路由：

- 满足 S 的质量、稳定性和低重复要求后，其 FR/PV/HF/OB 原子成员进入
  `self_factor_composite`；
- INT 等已经包含多个底层信号的复合候选不递归进入家族等权组合，避免同一信号
  重复计权；它们只有通过 S 后才可继续完成 I/T 评价；
- `I candidates = frozen_S`，`T candidates = frozen_I`。任一层未通过，后续层不再
  读取该候选。

候选池冻结之后，最终路线选择使用 `competition_score_proxy.py` 计算本地 J。J 的本地参考池是
`all36 + J baseline candidates`。all36 是公开基础坐标；J baseline candidates 是候选模块显式标记 `INCLUDE_IN_J_BASELINE = True` 的我方候选，用于模拟本队
已知拥挤环境，判断一条最终路线相对“公开库 + 自己库上已有信号”是否仍有线性
贡献。自研原子因子不得混写入 all36 目录。对任一路由输出 z：

```text
A(z) = 0.25 × [Pct(IC_mean) + Pct(IC_IR) + Pct(SR) + Pct(Stress)]
B(z) = Pct(mean(abs(w_z)) / (std(abs(w_z)) + epsilon))
J(z) = 0.3 × A(z) + 0.7 × B(z)
```

A 的四个百分位均把一个路由输出插入固定 `all36 + J baseline candidates` 分布后按平均秩
计算。最终路线选择使用 `all36 + J baseline candidates + 一个 route_output`，使用截面
z-score 标签、60 日窗口、20 日步长和允许正负系数的评分 Elastic Net；baseline
与 augmented 必须分别替换同一个路由槽，禁止同时进入贡献模型。构造 I 路由的
正系数 Elastic Net 与计算 B 的无符号限制 Elastic Net 是两个不同模型合同。

最终三条提交路线另做一次有界拥挤检查：各路线先单独相对
`all36 + J baseline candidates` 计分，再把已经冻结方向的三条兄弟路线一次性放入同一个
`all36 + J baseline candidates + sibling_routes` Elastic Net。
这只用于检查我方路线之间可观察的相互替代，是未知全局拥挤程度的下界，不把我方
历史或当前路线冒充全体参赛者历史，也不枚举三套完整压力网格。最终稳健 J 取
2023、2024、两年合并基础场景与一次兄弟路线拥挤场景中的最低值。

`alpha、l1_ratio`、压力期划分等未由公告完整披露的细节必须在报告中标记为
“本地代理假设”。这是评分器参数误差，与“看不到平台全局候选池”的参考池误差
属于两类不同不确定性；两者都不得写成官方精确复刻。

I 与 T 的最终联合模型使用相同的 60 日训练、20 日样本外窗口。每个窗口的训练和
预测只能使用该窗口开始前的数据。S/I/T 轻量准入在开发期完成后冻结成员，随后运行
2023、2024。缓存按实际输入内容寻址：新增或修改候选只使包含该候选的结果失效；
标签、screened15、预处理或模型配置变化时，对应结果全部失效。冻结成员禁止被
静默替换或删除。

不保留没有明确后续训练动作的 `watch`、`development_survivor` 或风险变量状态。

## 6. screened15 增量评价（I）

核心函数：

```text
factorlib_regularized_incremental_validation
factorlib_regularized_incremental_batch_validation
```

位置：`src/bigalpha2026/evaluation.py`。

I 的候选评价、联合池确认、冻结池晋级和缓存状态写入统一由
`src/bigalpha2026/incremental_admission.py` 的
`run_incremental_admission` 编排；上述 `evaluation.py` 函数只负责数值验证原语。

流程：

1. 复用所有候选共用的股票日、标签、冻结 screened15 和模型参数；I 评价只读取
   2019—2022 开发期，不用 2023 或 2024 决定模型成员。
2. screened15 已经用 2019—2021 从公开 36 因子中筛选并冻结；不得重复筛选，
   也不得用 2022、2023 或 2024 修改成员。
3. I 准入本身不做逐候选滚动训练；60/20 只属于最终 Elastic Net 的 walk-forward
   训练与评价合同。
4. 每个 pending 候选先做严格 I entry gate：质量合格后，原始 Rank IC 或相对
   `screened15 + frozen_I` 的残差 Rank IC 必须达到 `0.005`；同时最大绝对 Rank
   相关性必须不高于 `0.50`，除非残差 Rank IC 已达到 `0.005`。
5. 所有候选使用相同开发日历；特征与标签均按日转换为中心化截面百分位秩，
   缺失或无截面离散度的候选值按中性值 0 处理，禁止通过各自活跃日期改变
   baseline 样本。
6. 所有公开因子和自研因子已按 2019—2022 冻结方向统一为
   “值越大越好”，因此 Elastic Net 固定使用非负系数，禁止短窗口噪声把已验证
   信号反向使用。
7. 通过 entry gate 的候选直接进入 `frozen_I`。准入阶段不再训练逐因子 Elastic Net，也不计算本地 J；原因是本地 J 与官方分数
   不完全一致，强制要求本地分数增加可能误删官方可得分因子。
8. 轻量口径保留条件前向相关报告字段用于兼容旧报表，但不再用条件前向 J 否决。
9. `frozen_I` 只由 entry gate 决定，更新时仍记录候选指纹、标签指纹和评估合同；
   因子取值或合同变化时走新缓存版本，不静默复用旧 frozen state。
10. 最终 Elastic Net 的模型特征只读取 `frozen_I`；`screened15` 用于构造残差
    训练目标，并在残差预测后作为冻结基线完整加回，但不进入模型特征。
    2023、2024 的表现不得反向修改候选级 I、联合确认或冻结成员。

I entry gate 当前口径：

- `coverage >= 0.90`；
- 至少 120 个有截面离散度的开发日；
- 原始 Rank IC 均值不低于 `0.005`，或对 `screened15 + frozen_I` 做日内秩残差化
  后的 residual Rank IC 不低于 `0.005`；
- 与 `screened15 + frozen_I` 的最大绝对 Rank 相关性不高于 `0.50`，或 residual
  Rank IC 已达到 `0.005`。

I 的定位是“线性增量”：它不要求候选单独足够强，但必须证明有可被 Elastic Net
使用的线性或残差信息。

I 的正式准入门槛就是上述 entry gate。权重和相关性可以在最终模型训练后报告；准入阶段不跑本地 J。

I 缓存分成两层：

- `validation`：保存 entry/诊断摘要，不拥有正式成员状态；
- `frozen`：保存 entry 通过后的 `frozen_I`。

缓存键必须包含候选实际取值、screened15 实际取值、标签、有效日期、2019—2022
时间切分、60/20 窗口、预处理和 Elastic Net 参数。`factor_pool_incremental.csv`
只是审计报告，禁止再用“候选名称 + 协议文字”充当缓存。已冻结因子的值、可用性
或评价合同变化时必须直接停止，恢复冻结版本或把修改登记为新候选重新验证。
命令入口仍为 `scripts/run_combinations.py`；它只准备共享输入并调用
`run_incremental_admission`，精确缓存自动复用，
`--resume-incremental` 仅为向后兼容。

I 只评价通过 S 的候选；LightGBM 使用第 7 节的正交准入确认 T，并且只评价
通过 I 的候选。

技术门槛未通过时直接标记 `technical_reject`。S/I/T 的前置规则不同：S 看质量、强度、稳定性和低冗余，I 看线性或残差信息，T 看最低质量和正交性。`all36 + J baseline candidates`
只在候选池冻结后的最终路线选择中使用。

这是官方评分的代理检查，不等于真实比赛分数。

## 7. LightGBM 增量评价（T）

T 的准入模块为 `src/bigalpha2026/tree_admission.py`，核心入口为
`run_tree_admission`。模块拥有完整池确认、冻结池晋级及缓存状态；
`scripts/run_combinations.py` 只负责准备共享输入并调用该入口。

T 的输入池固定为 `frozen_I`，只使用 2019—2022 开发期。满足最低质量门槛后，候选
按“与当前基准的最大绝对 Rank 相关性从低到高、候选 ID 打破并列”的确定性顺序检查。
当前基准初始为 `screened15 + 已冻结 frozen_T`；同一次运行中每接纳一个候选，就立即
加入当前基准，后续候选必须同时与它保持正交。逐因子 LightGBM、Rank IC 和条件前向
J 增量不作为准入步骤；准入阶段不跑本地 J。
所有输入按相同股票日、标签、60 日训练、20 日 OOS 和中心化截面百分位秩处理，
并固定浅层 LightGBM 参数和正单调约束。

T 的正式准入门槛：

- 至少 120 个有截面离散度的开发日；
- 与 `screened15 + 已冻结 frozen_T + 本批已接纳 T` 的最大绝对 Rank 相关性不高于
  `0.35`。

Rank IC、正 J 窗口比例、正 J 年份数、分裂次数、gain importance 与 SHAP 只作解释
和诊断，不作准入条件。

T 缓存分成两层：

- `validation`：保存正交检查状态，不拥有正式模型状态；
- `frozen`：保存正交通过后的 T 池状态及最终 LightGBM 预测。

缓存键必须包含实际特征值指纹、标签指纹、特征集合、评价年份、60/20 窗口及
LightGBM 参数。新增或修改候选只使包含该候选的最终联合模型失效；完全不含它的
`screened15` 或旧冻结模型可以继续复用。已冻结因子的取值、可用性、标签口径或
模型配置发生变化时必须直接停止，禁止静默删除或重建冻结池；应恢复冻结版本，
或把修改注册为新候选并重新走完整 T 准入流程。

固定路由为：

- 全部合格候选先跑 S；
- `S=通过`：原子因子可进入规则复合，同时进入 I 候选池；
- `I=通过`：进入联合 Elastic Net，同时进入 T 候选池；
- `T=通过`：进入联合 LightGBM；
- 任一层未通过：停止该候选的后续评价。

高频或盘口候选不足 120 个开发有效日时不运行 T，也不得据此声称树模型无增量；
补齐连续日频聚合后再评价。

## 8. 组合与模型训练

通用实现：

- `src/bigalpha2026/combinations.py`
- `scripts/run_combinations.py`

`combinations.py` 只保留准入后的组合、Elastic Net 和 LightGBM 训练原语；
S/I/T 的流程与状态分别归属三个 admission 模块，候选池按 `S -> I -> T`
逐层收缩。组合训练仍固定为三条彼此隔离的管线：

1. `self_factor_composite`：所有获准自研因子按数据族内等权、族间等权合成；
2. `joint_elastic_net`：冻结的 15 个公开因子与 `I=通过` 的自研因子联合进入
   Elastic Net；
3. `joint_lightgbm`：冻结的 15 个公开因子与 `T=通过` 的自研因子进入浅层
   LightGBM。

三条管线不共享拟合后的权重、模型或预测，只共享输入面板、时间切分、缺失值处理
和评价口径。每条管线分别输出指标和准入结论，禁止在同一个通用
“输入池 × 模型”循环中临时增加第四种方案。

其中“获准”的含义按管线隔离：

- `self_factor_composite` 只读取 `S=通过` 的候选；
- `joint_elastic_net` 只读取 `I=通过` 的候选，并与冻结的 15 个公开因子联合训练；
- `joint_lightgbm` 只读取 `T=通过` 的候选，并与冻结的 15 个公开因子联合训练；
`factorlib_screened` 的模型结果已经在第 6 节 I 评价中计算并缓存，组合阶段仅将
其作为固定参照，不再作为第三个方案重复训练。只有模型结构、训练窗口或样本合同
改变时，才必须同步重算对应的公开库参照，否则不能声称联合模型存在增量。

冻结的 15 个公开因子为：

```text
amount, atr_14, bias_20, cci_14, float_market_cap,
kdj_d_9_3_3, macd_diff_12_26_9, macd_hist_12_26_9, momentum_5,
net_profit_rate_ttm, netflow_amount_rate_main, total_market_cap,
turn, volatility_5, volume
```

代码中的完整列名带 `factorlib__` 前缀，唯一事实来源为
`research_policy.py` 的 `FROZEN_FACTORLIB_SCREENED_FEATURES`。

不得把 `+FR-005`、`+FR-004` 之类的逐因子增强模型当作正式组合方案。
逐因子增量模型不在准入阶段运行。正式组合只读取已经冻结的
`frozen_I` 或 `frozen_T` 成员。

组合顺序：

1. 公开库先做覆盖率、常数、有限值和主键检查；
2. 读取已经用 2019—2021 冻结的 `factorlib_screened`；
3. 复用自研候选的 S 结果，并完成第 6 节 I 与第 7 节 T 评价；
4. `S=通过` 的候选进入规则组合；`I=通过` 的候选进入 Elastic Net；
   `T=通过` 的候选进入 LightGBM；
5. 分别执行规则组合、联合 Elastic Net、联合 LightGBM 三条管线；
6. 两个联合模型均使用过去 60 个交易日训练、随后 20 个交易日 OOS 预测；
7. 两个联合模型的最终训练使用同一个样本入口：特征与标签都按交易日转换为
   中心化截面百分位秩，标签无效的行删除，零方差或缺失特征填为中性值 0；
   Elastic Net 固定非负系数，LightGBM 固定正单调约束；I/T 准入也使用完全相同
   的模型合同，每个滚动窗输出并保存系数或预测；
8. 三条路线先在合并的 2023—2024 上分别比较 `z` 与 `-z` 的 J，冻结 J 更高的
   单一方向；不得同时保留或提交两个方向；
9. 对冻结方向后的三条路线分别计算 2023 J、2024 J、两年合并基础 J，并用一次
   联合 Elastic Net 计算兄弟路线拥挤 J；按四者最低值 `robust_score_proxy`
   从高到低冻结提交顺序，若完全并列再比较两年合并基础 J，最后优先简单的
   `self_factor_composite`。Rank IC、t 值、可交易子集和稳定性只报告风险，
   不得否决或覆盖更高 J 的路线。

训练代码在本地读取 AIStudio 核验快照、执行、测试和版本化，冻结赢家后才同步到
AIStudio 做短窗验收与提交。
新增因子只能扩充注册表和特征列，不能复制训练分支。所有模型共享股票池左表、
时间切分、缺失值和评价口径；平台只向本地导出报告、重要性和预测结果。

在没有真实快照时运行
`PYTHONPATH=src conda run --no-capture-output -n quant python scripts/run_combinations.py --check`
只使用合成数据验收动态列、左连接、中性填充和三条隔离管线的输出合同，不产生
有效性结论；读取已核验真实快照的完整运行属于正式研究结果。

真实快照检查和训练入口：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py \
  --data-dir /Users/yuanye/Projects/bigquant/data \
  --reports-dir /Users/yuanye/Projects/bigquant/reports \
  --check-files
```

`--check-files` 只打印并保存快照合同，不训练。完整训练去掉 `--check-files`。需要
强制重算所有 I/T 因子时增加 `--refresh-incremental-cache --refresh-tree-cache`；
只重算单个候选时使用 `--refresh-incremental-candidate CANDIDATE_ID` 和
`--refresh-tree-candidate CANDIDATE_ID`。

每条最终管线都必须比较：

- 2023、2024、两年合并基础场景和兄弟路线拥挤场景的 J；
- 正负方向的 J 及最终冻结方向；
- 2023、2024 各自的 Rank IC、t 值和可交易子集结果；
- 两年中较差的 Rank IC 与两年均值；
- 可交易子集；
- 模型权重或特征重要性；
- 权重或特征重要性的时间稳定性。

模型预测最终仍须转换为日度截面因子。方向由合并验证期 J 的正负比较冻结；传统
“值越大代表预期收益越高”、样本外 IC 和简单等权比较只作为解释与风险诊断，
不能替代比赛 J 排名。

## 9. 冻结与提交

提交前必须冻结：

- 数据字段和可用时点；
- 缺失值和异常值处理；
- 因子方向、窗口和参数；
- 组合成员与权重；
- 模型结构、训练区间和重训规则；
- 输出 Notebook 版本。

提交准入：

- 三列合同、覆盖率和防泄漏通过；
- 开发期以及 2023、2024 两个验证期结果完整；
- 最终路线的基础 J、拥挤 J、正负方向和稳健 J 报告完整；
- Rank IC、可交易子集和稳定性诊断完整，但不作为高 J 路线的否决门槛；
- Notebook 可在 AIStudio 用任意短日期运行；
- 用户明确授权提交。

比赛网页最多选择两个私榜候选。未完成冻结的实验候选不得占用提交名额。

## 10. 固定产物

每轮研究至少保存：

```text
reports/
├── latest/
│   ├── factor_pool_check.json
│   ├── factor_pool_admission.csv
│   ├── combination_summary.csv
│   └── factor_pool_decisions.json
├── first_round/
│   ├── first_round_technical.csv
│   ├── first_round_metrics.csv
│   ├── first_round_stability.csv
│   └── first_round_decisions.json
├── routes/
│   ├── competition_J_reference_directions.csv
│   ├── factor_pool_screening.csv
│   ├── single_factor_route_admission.csv
│   ├── single_factor_route_promotion.csv
│   ├── factor_pool_incremental.csv
│   ├── incremental_factorwise_admission.csv
│   ├── incremental_factorwise_promotion.csv
│   ├── incremental_pool_promotion.csv
│   ├── tree_factor_admission.csv
│   ├── tree_factor_incremental.csv
│   ├── tree_factorwise_admission.csv
│   ├── tree_factorwise_importance.csv
│   ├── tree_factorwise_promotion.csv
│   ├── self_factor_composite_metrics.csv
│   ├── joint_elastic_net_metrics.csv
│   ├── joint_elastic_net_weights.csv
│   └── joint_lightgbm_metrics.csv
└── diagnostics/

artifacts/frozen/
└── int_001.json             # 已提交历史版本，因子池实验不得覆盖
```

失败结果同样保留，防止重复试验和事后改写研究历史。
候选相关性矩阵是可选诊断，不参与 S/I 准入；使用
`--skip-correlations` 时不要求生成或提交。
