# 队友与 AI 因子开发使用手册

这份文档是新成员进入仓库后的操作入口。开始前依次阅读：

1. `docs/team_workflow.md`：怎么协作和运行；
2. `docs/candidate_registry.md`：哪些候选已经存在；
3. `docs/data_contract.md`：允许使用哪些字段以及时间含义；
4. `docs/factor_research_plan.md`：如何评价和晋级。

## 0. 首次登录与获取项目

先用本人 BigQuant 账号在浏览器中登录：

- 比赛页面：
  `https://bigquant.com/square/competition/76ad3f56-ec2b-431a-890e-139a7f4bbcba`
- AIStudio：
  `https://bigquant.com/aistudio/landing?aistudio_version=300`

确认可以进入当前团队、查看提交记录、打开 AIStudio Notebook，并访问比赛提供的
`instruments、bar1m、financial、exposure、factorlib`。账号和密码由本人手动
输入，不交给 AI；登录完成后，AI 可以继续使用当前浏览器完成取数、验证和提交。

仓库地址：

```text
https://github.com/yeyeyuan23/bigquant
```

同队成员加入原仓库 Collaborator 后直接 clone，不需要 fork：

```bash
git clone https://github.com/yeyeyuan23/bigquant.git
cd bigquant
conda run --no-capture-output -n quant python -m pytest -q
```

## 1. 工程地图

```text
src/bigalpha2026/
├── candidates/
│   ├── pv/              # 日频价量
│   ├── hf/              # 分钟成交聚合
│   ├── ob/              # 五档盘口聚合
│   ├── fr/              # PIT 财务披露
│   └── composite/       # 已冻结的跨类组合或模型
├── evaluation.py        # 单因子与公开因子库增量评价
├── combinations.py      # 固定组合和前推树模型
├── factorlib.py         # 公开基础因子库字段合同
└── research_policy.py   # 候选、月份和准入门槛

scripts/
├── run_first_round.py   # 八个基础候选的完整评价入口
├── run_combinations.py  # 组合、Elastic Net 和树模型入口
└── build_submission_notebook.py

tests/                   # 单元测试
reports/                 # 评价与冻结结果
docs/                    # 合同、登记和协作规则
```

## 2. 开始协作前

### 人工确认

在群里先声明准备研究的编号和类别，例如：

```text
我认领 HF-003：尾盘成交集中后的价格恢复。
```

不要两个人同时使用同一个编号。认领后先在
`docs/candidate_registry.md` 按模板登记，再写代码。

### Git 协作

每个候选使用独立分支和独立提交：

```bash
git switch -c factor/hf-003
```

完成后向原仓库提交 MR/PR，不直接修改或合并 `main`。一次提交只处理一个
候选，避免同时改动其他人的候选文件。不要提交：

- 本地原始数据；
- AIStudio 下载缓存；
- Notebook 临时输出；
- 密钥、Cookie 或账号信息；
- 与本候选无关的格式化修改。

## 3. 新增一个基础因子

### 第一步：选择类别

| 使用信息 | 目录 | 编号 |
|---|---|---|
| 日频价格和成交 | `candidates/pv/` | `PV-xxx` |
| 分钟成交和价格路径 | `candidates/hf/` | `HF-xxx` |
| 五档报价和深度 | `candidates/ob/` | `OB-xxx` |
| 财务披露 | `candidates/fr/` | `FR-xxx` |

同时使用两个以上类别时，先分别确认基础组件有效。跨类候选只有进入组合阶段后
才能放入 `candidates/composite/`。

### 第二步：登记候选

登记内容必须在查看结果前写定：

- 经济机制；
- 使用字段；
- 信息最早可用时间；
- 因子正方向；
- 预测周期；
- 可能失败的市场状态；
- 与现有因子或公开因子库的重复风险。

如果只是把现有因子的窗口从 5 改成 6，不能自动算作新机制。

### 第三步：实现

文件名和编号一致，例如：

```text
src/bigalpha2026/candidates/hf/hf_003.py
```

实现时遵守：

1. 数据读取、组件计算和最终输出分离；
2. 不在候选模块里读取未来标签；
3. 滚动窗口只能使用当日及以前的数据；
4. 输出严格为 `date、instrument、factor`；
5. 因子值越大代表预期下一期收益越高；
6. 对缺失值、停牌、午休、盘口空档或财务披露时点做显式处理；
7. 在对应类别的 `__init__.py` 中导出公开函数。

### 第四步：添加测试

测试文件：

```text
tests/test_hf_003.py
```

至少检查：

- 缺少必要列时明确报错；
- 输出列和主键正确；
- 停牌或无数据股票仍符合股票池合同；
- 修改未来输入不会改变过去输出；
- 因子方向与登记一致；
- 极端值不会产生无穷数。

运行：

```bash
conda run --no-capture-output -n quant python -m pytest -q
```

所有测试通过后才能进入 AIStudio 真实数据验收。

## 4. AIStudio 数据验收

AIStudio 只负责真实数据查询和共享面板生成，不在 Notebook 中临时改变因子方向、
评价门槛或组合权重。

顺序：

1. 用 2—5 个真实交易日和少量股票查询必要字段；
2. 核对字段类型、分钟增量、交易时段、财务 PIT 和盘口空档；
3. 扩到比赛股票池，检查每日行数、覆盖率和重复键；
4. 在平台侧聚合成所属数据族的日级共享面板；
5. 保存 Parquet 后重新读回，核对行数和字段；
6. 使用该共享面板运行候选输出轻检；
7. 记录数据合同变化，不把“可以运行”写成“因子有效”。

原始分钟和财务宽表留在 AIStudio。只有符合 `data_contract.md` 的共享面板进入
本地评价。

如果新因子需要现有 HF/OB 面板没有的新分钟逻辑，直接使用平台传入的
`datasources["bar1m"]`。必须先做短窗验证，再聚合为以
`date、instrument` 为主键的日级通用组件。分钟收益不能跨午休；
`volume、amount、deal_number` 按分钟增量处理；五档盘口只有价格和数量均为正
的档位有效。

## 5. 本地数据包同步

比赛数据不进入 Git。需要在赛事规则允许的同队成员之间同步当前聚合研究面板时，
在仓库根目录执行：

```bash
mkdir -p data/transfers

tar \
  --exclude='data/raw' \
  --exclude='data/transfers' \
  --exclude='*.tmp' \
  --exclude='*.partial' \
  -czf data/transfers/bigalpha_research_data_20260726.tar.gz \
  data

shasum -a 256 \
  data/transfers/bigalpha_research_data_20260726.tar.gz \
  > data/transfers/bigalpha_research_data_20260726.tar.gz.sha256
```

发送压缩包和同名 `.sha256` 文件。接收方将二者放到仓库的
`data/transfers/` 后，先在仓库根目录验证：

```bash
shasum -a 256 -c \
  data/transfers/bigalpha_research_data_20260726.tar.gz.sha256
```

校验显示 `OK` 后解压：

```bash
tar -xzf \
  data/transfers/bigalpha_research_data_20260726.tar.gz \
  -C .
```

最后运行：

```bash
conda run --no-capture-output -n quant python -m pytest -q
```

当前包只包含聚合后的 `universe、features、exposures、labels、factors` 和
manifests，不包含原始分钟成交或盘口快照。`data/transfers/` 已被
`.gitignore` 排除，禁止强制加入 Git。

如果数据内容发生变化，使用当天日期生成新包，不覆盖旧包；双方通过 SHA-256
确认使用的是同一份数据快照。

当前团队基础数据包：

```text
bigalpha_research_data_20260726.tar.gz
bigalpha_research_data_20260726.tar.gz.sha256
SHA-256:
4c2b6a88a9aae109bbb47fd4b46cfd77cc2c2a05adfb1db627ef0e5aab8f4fc6
```

如果候选引入当前包中没有的新日级组件，开发者必须同时交付：

1. 生成组件的 AIStudio 查询或 Notebook；
2. 更新后的 `docs/data_contract.md`；
3. 新增日级 Parquet；
4. 对应 manifest；
5. 增量压缩包和 SHA-256。

增量包示例：

```text
bigalpha_data_delta_HF-003_20260726.tar.gz
bigalpha_data_delta_HF-003_20260726.tar.gz.sha256
```

只打包新增组件，不重新发送完整数据包。队长解压后必须能运行同一评价代码得到
一致结果。只提供 IC 截图或模型结果、没有聚合代码和增量数据的候选不可复现，
不能作为最终版本接收。

## 6. 评价新因子

### 基础评价

把候选接入 `scripts/run_first_round.py`，然后运行：

```bash
conda run --no-capture-output -n quant python scripts/run_first_round.py
```

查看：

```text
reports/first_round_technical.csv
reports/first_round_metrics.csv
reports/first_round_stability.csv
reports/first_round_costs.csv
reports/first_round_correlations.csv
reports/first_round_decisions.json
```

不要只看一个 IC。必须同时看方向稳定性、分组单调性、可交易子集、中性化、
换手和成本。

### 公开因子库增量

候选完成基础评价后，调用：

```python
factorlib_regularized_incremental_validation(...)
```

比较“仅公开因子库”和“公开因子库 + 候选”的严格前推 OOS 结果。记录：

- Rank IC 增量；
- 正增量日期和窗口比例；
- 候选非零权重比例；
- 权重方向稳定性；
- 与公开因子的最大 Rank 相关。

如果单因子弱但增量稳定，可登记为 `diversifier`；如果单因子和增量都不稳定，
标记 `rejected`。

## 7. 进入组合层

只有基础候选完成单因子评价后，才能修改：

```text
src/bigalpha2026/combinations.py
scripts/run_combinations.py
```

运行：

```bash
PYTHONPATH=src conda run --no-capture-output -n quant python scripts/run_combinations.py
```

正式只比较两组输入：公开库全部 36 因子，以及“统一筛选后的公开因子 + 首轮准入
的自研因子”。不再单列“仅筛选公开因子”组。每组先在数据族内等权、再对数据族
等权，之后比较 Elastic Net 和浅层 LightGBM；XGBoost 只作对照。
复杂模型只有在 2022 选择期和 2023 确认期都稳定超过简单基准时才能保留。

组合脚本必须从登记表或标准化因子文件动态读取特征列，禁止为每个新增因子手工增加
一套组合分支。统一以历史股票池为左表；特征转为日度截面秩后，缺失值填为截面中性
值 `0`，不得用全部特征的交集缩小股票池。可先运行
`scripts/run_combinations.py --check`，只验收动态列、两组成员和缺失值合同，
不训练模型。

不要把一次训练产生的模型直接放进 `composite/`。必须先冻结：

- 输入组件；
- 训练区间；
- 超参数；
- 重训频率；
- 缺失值处理；
- 推理接口；
- 输出方向。

## 8. 比赛提交与代码交付

候选完成本地评价和 AIStudio 真实数据验证、结果值得提交时，同队成员及其 AI
可以直接上传比赛，不需要再次询问队长。

提交前必须：

- 冻结代码、参数和 Notebook；
- 用 2—5 个真实交易日验证；
- 输出严格为 `date、instrument、factor`；
- 主键无重复，因子值有限，覆盖率合格；
- 接受平台传入的任意日期范围；
- 不读取本地文件、不访问外部网络、不写死评价区间。

上传后只需告诉队长：

- Notebook 文件名和 `candidate_id`；
- 提交前、提交后的团队排名；
- 排名提升了多少；
- 是否刷新队内最佳结果；
- 记录排名的时间。

不要为了刷新榜单反复提交只有细小参数差异的同一个因子。

完成后把最终代码推到候选分支并创建 MR/PR：

```bash
git status
git add <本候选相关文件>
git commit -m "Add HF-003 candidate"
git push -u origin factor/hf-003
```

MR/PR 简单说明：

- 因子机制；
- 测试和 AIStudio 验收是否通过；
- 是否已经上传及排名提升；
- 是否包含新数据依赖和增量包。

不要自行合并 `main`，最终版本由主仓库统一保留。

## 9. 合并前检查

- 候选编号没有冲突；
- 登记内容与代码方向一致；
- 只修改本候选和必要的公共入口；
- 完整测试通过；
- 没有数据文件、密钥或 Notebook 输出；
- AIStudio 验收和本地有效性结论被明确区分；
- 失败结果已记录；
- 新数据依赖已经交付生成代码、manifest 和增量包；
- 最终代码已创建 MR/PR。
