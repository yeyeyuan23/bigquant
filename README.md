# BigAlpha 2026 因子研究工程

本仓库用于管理可复现的因子研究流程：候选登记、数据合同、单因子评价、公开因子库
增量评价、组合训练、冻结和比赛提交。README 只保留稳定入口；候选状态和实验结果
不在这里重复维护。

## 从这里开始

- 团队开发与交付：[协作手册](docs/team_workflow.md)
- 研究流程、时间切分和准入门槛：[因子研究与评价流程](docs/factor_research_plan.md)
- 字段、主键和可用时点：[数据合同](docs/data_contract.md)
- 报告目录地图和 CSV 字段读法：[Reports 说明](reports/README.md)
- 候选定义、方向和当前状态：[候选登记表](docs/candidate_registry.md)
- 当前可执行路由：`reports/latest/factor_pool_admission.csv`
- 当前三管线结果和成员：`reports/latest/combination_summary.csv` 与
  `reports/latest/factor_pool_decisions.json`
- 已提交版本的不可变配置：`artifacts/frozen/`

文档职责保持分离：研究规则只在执行计划中定义，数据口径只在数据合同中定义，
候选状态只在登记表中维护。代码侧的可执行研究配置以
`src/bigalpha2026/research_policy.py` 为准。

## 阅读顺序

如果目的是理解整个工程，按这个顺序读：

1. `README.md`：仓库边界、目录和稳定入口。
2. `docs/data_contract.md`：比赛允许的数据表、主键、时点和本地落盘格式。
3. `docs/candidate_registry.md`：候选编号、机制和当前候选池版本。
4. `docs/factor_research_plan.md`：S/I/T 准入、J 代理、训练窗口、冻结规则。
5. `reports/README.md`：每个稳定报告文件的目录位置和字段含义。
6. `docs/team_workflow.md`：实际执行命令、数据同步、AIStudio 验收和提交协作。
7. `scripts/run_first_round.py`：候选长表和单因子报告生成入口。
8. `scripts/run_combinations.py`：S/I/T、三条组合管线和最终报告生成入口。

## 固定研究边界

- 本地仓库负责候选和模型代码、评价与组合训练、研究规则、测试及版本控制。
- AIStudio 负责真实数据查询、字段和快照核验，以及冻结版本的最终短窗验收。
- 比赛网页只用于提交已经冻结并通过验收的版本，以及记录平台反馈。
- 正式研究结果必须来自通过数据合同和 manifest 核验的真实快照；合成数据只用于
  接口和结构检查。
- 原始比赛数据、分钟数据和队内研究快照不进入 Git。
- 平台冒烟提交与正式研究准入分开记录，公榜反馈不作为无约束调参数据。

## 固定工程合同

- 基础因子分为 `PV、HF、OB、FR` 四类，跨类方案登记为 `composite`。
- 候选必须先登记机制、方向、字段和可用时点，再实现和评价。
- 未登记的候选不进入代码库；失败候选保留状态和原因，避免重复试验。
- 候选入口 `main()` 只返回 `date、instrument、factor`，且值越大代表预期收益越高。
- 每个提交 Notebook 只包含一个因子入口，不在平台端临时修改研究规则。

固定流程为：

```text
登记候选
→ 实现并测试
→ 核验真实数据快照
→ 单因子评价（S）
→ 相对冻结公开因子库的增量评价（I）
→ LightGBM 增量评价（T）
→ 按 S/I/T 结果路由到隔离的组合管线
→ 冻结代码、数据合同、成员、参数和 Git 版本
→ AIStudio 短窗验收
→ 比赛提交
```

组合层只保留三条独立管线：

- `self_factor_composite`
- `joint_elastic_net`
- `joint_lightgbm`

具体时间区间、门槛、模型参数和固定产物清单统一见
[因子研究与评价流程](docs/factor_research_plan.md)，不在 README 复制。

## 本地执行入口

只检查代码结构和三条组合管线的合成数据合同：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py --check
```

只检查真实快照文件和 manifest，不训练模型：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py \
  --data-dir /Users/yuanye/Projects/bigquant/data \
  --reports-dir /Users/yuanye/Projects/bigquant/reports \
  --check-files
```

完整重跑 S/I/T 和三条组合管线：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py \
  --data-dir /Users/yuanye/Projects/bigquant/data \
  --reports-dir /Users/yuanye/Projects/bigquant/reports
```

强制重算所有 I/T 因子的训练增量：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py \
  --data-dir /Users/yuanye/Projects/bigquant/data \
  --reports-dir /Users/yuanye/Projects/bigquant/reports \
  --refresh-incremental-cache \
  --refresh-tree-cache
```

只重算某个新增或修改因子的 I/T 增量：

```bash
cd /Users/yuanye/Projects/bigquant
PYTHONPYCACHEPREFIX=/tmp/bigquant-pycache conda run --no-capture-output -n quant \
  python /Users/yuanye/Projects/bigquant/scripts/run_combinations.py \
  --data-dir /Users/yuanye/Projects/bigquant/data \
  --reports-dir /Users/yuanye/Projects/bigquant/reports \
  --refresh-incremental-candidate CANDIDATE_ID \
  --refresh-tree-candidate CANDIDATE_ID
```

其中 `--check-files` 打印真实快照合同后立即退出，不会训练；不带
`--check-files` 时，打印同一份合同后会继续训练。

## 工程结构

```text
src/bigalpha2026/
├── candidates/           # 已登记的基础因子和跨类方案
├── evaluation.py         # 评价指标与滚动验证原语
├── competition_score_proxy.py # all36+self_library 本地 J 代理、方向选择与联合拥挤 J
├── single_factor_admission.py  # S 单因子准入
├── incremental_admission.py    # I Elastic Net 增量准入
├── tree_admission.py            # T LightGBM 增量准入
├── combinations.py       # 准入后的最终组合与模型训练
├── factor_pool.py        # 公开库和自研因子的动态特征池
├── factorlib.py          # 公开因子库字段合同
└── research_policy.py    # 可执行研究配置

scripts/
├── run_first_round.py
├── run_combinations.py
└── build_submission_notebook.py

docs/                     # 流程、合同、登记和协作规则
reports/                  # 可复现的评价结果、当前路由和模型成员
artifacts/frozen/         # 已提交版本的不可变配置
tests/                    # 单元、接口和防泄漏测试
```

## 结果文件读法

- `reports/latest/factor_pool_check.json`：真实快照合同检查结果。
- `reports/first_round/first_round_*.csv/json`：候选单因子技术、指标和稳定性诊断。
- `reports/routes/incremental_factorwise_admission.csv`：I 中每个因子的 Elastic Net 个人
  增量、条件前向和冻结状态。
- `reports/routes/incremental_conditional_forward.csv`：I 条件前向每一步的 baseline、
  candidate 和配对 J 增量。
- `reports/routes/tree_factorwise_admission.csv`：T 中每个因子的 LightGBM 个人增量、
  条件前向和冻结状态。
- `reports/routes/tree_factorwise_importance.csv`：T 中每次 LightGBM 训练的 split/gain
  importance。
- `reports/routes/tree_factorwise_promotion.csv`：T 条件通过池的整体确认结果。
- `reports/latest/factor_pool_admission.csv`：每个候选最终进入 S/I/T 哪些路线。
- `reports/latest/combination_summary.csv`：三条最终路线的验证期表现和排序。
- `reports/latest/factor_pool_decisions.json`：机器可读的完整路由、模型和评分合同。

字段细节见 [Reports 说明](reports/README.md)。

## 本地环境

项目使用 Python 3.11 及 `quant` Conda 环境。首次 clone 后执行：

```bash
conda run --no-capture-output -n quant python -m pip install -e .
conda run --no-capture-output -n quant python -m pytest -q
```

只检查组合接口和动态列合同：

```bash
PYTHONPATH=src conda run --no-capture-output -n quant \
  python scripts/run_combinations.py --check
```

`--check` 使用合成数据，不产生因子有效性结论。真实评价的运行方式和所需数据包见
[协作手册](docs/team_workflow.md)。
