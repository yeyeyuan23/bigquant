# BigAlpha 2026 因子研究工程

当前工程只保留已登记候选、共享数据合同、评价规则和最小验证。后续工作以 [因子选择与评估执行计划](docs/factor_research_plan.md) 为唯一主流程。

## 当前有效内容

- `docs/factor_research_plan.md`：逐阶段研究、评价、冻结和提交计划；
- `docs/candidate_registry.md`：已登记候选、数据准入状态和最小实现记录；
- `docs/data_contract.md`：AIStudio 数据合同、字段和最小面板验收；
- `src/bigalpha2026/candidates/`：只包含已经登记并通过数据核验的候选实现；
- `src/bigalpha2026/evaluation.py`：与具体候选无关的通用评价工具；
- `src/bigalpha2026/research_policy.py`：因子池、门槛、代表月份和组合规则的本地唯一事实来源。

## 项目原则

- 原始分钟、盘口和财务全表只在网页 BigQuant AIStudio 查询；
- 同一数据族在 AIStudio 一次聚合为股票日 Parquet，并通过平台官方 UI 在下载配额内保存到本地；
- 因子池、候选定义、历史评价、组合和消融全部由本地版本化规则执行；
- 本地 Python 统一显式使用 `conda run --no-capture-output -n quant python`，不得调用系统 Python；
- Parquet 基础面板按 `date、instrument` 唯一，特征、财务事件、标签和因子输出分层保存；
- 未登记的候选不实现；
- 严格研究准入与“平台链路冒烟提交”分开记录；冒烟提交不得被表述为已通过交易准入；
- 每个提交 Notebook 只能包含一个因子入口；
- `main()` 只能返回 `date、instrument、factor` 三列。

## 当前结论

四类最小共享面板、2019—2023 年基础包、FR PIT 事件和 HF/OB 日频组件均已通过 AIStudio 与本地双重验收。八个基础候选的首轮评价已完成；随后按“先跑通最小组合链路”的独立目的，将 `FR-002` 与 `HF-001` 纳入组合准入观察池。

规则组合和机器学习基线已经完整比较：单因子、等权秩、75/25、训练期 IC 加权、Elastic Net、sklearn HGB、LightGBM 和 XGBoost 均走完同一扩展窗口评价接口。2022 选择期中等权秩组合 Rank IC 为 `0.00851`；XGBoost 达到 `0.01246`，2023 确认为 `0.01400`，显示出非线性增量。XGBoost 的日均多空毛收益与 20 bps 成本结果仍失败，换手也高于等权，因此登记为后续扩充因子池后的模型候选，不追溯替换已上传的 `INT-001`。

冻结的 `INT-001` 是 `FR-002/HF-001` 各 50% 的日度截面秩组合。它的 20 bps 成本后多空收益仍为负，因此不属于“可交易性已通过”的严格候选；本次仅作为平台端到端冒烟版本。2026-07-26，AIStudio 真实数据短窗返回 3,000 行、三列契约、无重复、覆盖率 100%，随后比赛网页确认提交成功；公榜分数为 `0.57237`。该分数只作为平台反馈，不用于追溯修改冻结权重。
