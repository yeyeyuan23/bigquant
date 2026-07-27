# 队友与 AI 因子开发使用手册

这份文档是新成员进入仓库后的操作入口。开始前依次阅读：

1. `docs/team_workflow.md`：怎么协作和运行；
2. `docs/factor_research_plan.md`：环境分工、研究流程、评价和晋级；
3. `docs/candidate_registry.md`：哪些候选已经存在；
4. `docs/data_contract.md`：允许使用哪些字段以及时间含义。

## 发给队友 AI 的启动 Prompt

```text
请打开比赛页面和 AIStudio，让我先完成登录。然后 clone
https://github.com/yeyeyuan23/bigquant，完整阅读 docs/team_workflow.md，
并严格按其中权限边界开发候选因子、测试并在 AIStudio 做真实评价。
有效结果可以直接上传比赛；完成后 push 候选分支并创建 MR/PR。
如收到数据压缩包，按手册校验和解压；如因子需要新数据，按手册一并交付
可复现的取数代码、数据合同、manifest 和增量包。
```

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
conda run --no-capture-output -n quant python -m pip install -e .
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
├── evaluation.py        # 共用评价原语
├── single_factor_admission.py  # S 单因子准入
├── incremental_admission.py    # I Elastic Net 增量准入
├── tree_admission.py            # T LightGBM 增量准入
├── combinations.py      # 准入后的三条组合训练原语
├── factorlib.py         # 公开基础因子库字段合同
└── research_policy.py   # 候选、月份和准入门槛

scripts/
├── run_first_round.py   # 单因子评价代码
├── run_combinations.py  # 组合与模型训练代码
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

### 修改权限

队友的候选分支只允许修改：

- 自己认领的 `src/bigalpha2026/candidates/<类别>/` 候选文件；
- 对应类别的 `__init__.py` 导出；
- `docs/candidate_registry.md` 中该候选的登记；
- 该候选对应的测试和小型评价报告；
- 新数据确有必要时，对应的取数 Notebook、`docs/data_contract.md`、manifest
  和增量包说明。

以下内容只读，不得在候选 MR/PR 中修改：

- `src/bigalpha2026/evaluation.py` 和通用评价标准；
- `src/bigalpha2026/combinations.py`、`scripts/run_combinations.py` 和组合训练逻辑；
- `src/bigalpha2026/research_policy.py` 中的时间切分与准入门槛；
- `src/bigalpha2026/candidates/composite/` 和 `artifacts/frozen/` 中的冻结版本；
- `main` 分支。

公共代码和研究规则的调整使用独立 MR/PR，由主仓库负责人统一处理；候选 MR/PR
只包含 `candidate` 范围内的实现、测试和登记信息。队友可以运行公共评价和组合
代码，但不能修改它们。

## 3. 新增一个基础因子

候选分类、登记字段、实现规范、技术门槛和评价准入统一见
`docs/factor_research_plan.md` 第 4—7 节；字段与时点只能引用
`docs/data_contract.md`，不要在本文件复制一套规则。

队友的操作顺序只有：

1. 在群里认领编号；
2. 在 `docs/candidate_registry.md` 登记；
3. 在自己的候选分支实现同编号模块和测试；
4. 本地完整测试通过；
5. 按第 4 节交给 AIStudio 做真实数据核验；
6. 有效结果可以上传比赛，随后提交 MR/PR。

文件名和编号必须一致，例如：

```text
src/bigalpha2026/candidates/hf/hf_003.py
tests/test_hf_003.py
```

运行：

```bash
conda run --no-capture-output -n quant python -m pytest -q
```

## 4. AIStudio 取数与最终验收

本地与 AIStudio 的职责、传入内容、回传产物和固定循环统一见
`docs/factor_research_plan.md` 第 2 节；真实表的连接和聚合规则见
`docs/data_contract.md`。

实际操作：

1. AIStudio 查询并核验股票池、标签、共享面板、风险暴露和冻结 screened15；
2. 将日级 Parquet、manifest 和查询版本同步到本地；
3. 本地运行 plan 规定的正式评价和三条组合管线；
4. 冻结赢家后，将其代码和 Git 版本同步回 AIStudio；
5. 用 2—5 个交易日验收三列输出，然后生成比赛提交。

禁止把合成数据测试写成有效性结论，也禁止在 Notebook 中临时修改 plan。使用
已核验真实快照的本地全量结果属于正式研究结果。

## 5. 可选本地数据包同步

比赛数据不进入 Git。经过核验的日级面板、screened15 和训练矩阵可在赛事规则允许
的同队成员之间同步；原始分钟数据仅在开发新分钟逻辑时按最小月份同步。需要同步时，
在仓库根目录执行：

```bash
mkdir -p data/transfers

tar \
  --exclude='data/raw' \
  --exclude='data/cache' \
  --exclude='data/transfers' \
  --exclude='*.tmp' \
  --exclude='*.partial' \
  --exclude='.DS_Store' \
  -czf data/transfers/bigalpha_research_data_v2.tar.gz \
  data

shasum -a 256 \
  data/transfers/bigalpha_research_data_v2.tar.gz \
  > data/transfers/bigalpha_research_data_v2.tar.gz.sha256
```

发送压缩包和同名 `.sha256` 文件。接收方将二者放到仓库的
`data/transfers/` 后，先在仓库根目录验证：

```bash
shasum -a 256 -c \
  data/transfers/bigalpha_research_data_v2.tar.gz.sha256
```

校验显示 `OK` 后解压：

```bash
tar -xzf \
  data/transfers/bigalpha_research_data_v2.tar.gz \
  -C .
```

最后运行：

```bash
conda run --no-capture-output -n quant python -m pytest -q
```

当前包只包含聚合后的 `universe、features、exposures、labels、factors` 和
manifests，不包含原始分钟成交或盘口快照。`data/transfers/` 已被
`.gitignore` 排除，禁止强制加入 Git。

如果数据内容发生变化，递增包版本号（`v2`、`v3`、`v4`），不覆盖旧包；即使
一天内生成多个版本也必须使用不同版本号。双方通过 SHA-256 确认使用的是同一份
数据快照。

当前团队基础数据包：

```text
bigalpha_research_data_v2.tar.gz
bigalpha_research_data_v2.tar.gz.sha256
SHA-256:
459bb593a33d817dd850a2e8465db01523a229e6c9c56886ff4aeaf6c4bc48bd
```

该快照已经包含最新的连续微观日级面板、冻结 screened15、标准候选长表
`data/factors/candidate_pool.parquet` 和全部 manifests；不包含原始分钟数据
或可由当前代码重算的 `data/cache/`。

如果候选引入当前包中没有的新日级组件，开发者必须同时交付：

1. 生成组件的 AIStudio 查询或 Notebook；
2. 更新后的 `docs/data_contract.md`；
3. 新增日级 Parquet；
4. 对应 manifest；
5. 增量压缩包和 SHA-256。

增量包示例：

```text
bigalpha_data_delta_HF-003_v1.tar.gz
bigalpha_data_delta_HF-003_v1.tar.gz.sha256
```

只打包新增组件，不重新发送完整数据包。队长解压后必须能运行同一评价代码得到
一致结果。只提供 IC 截图或模型结果、没有聚合代码和增量数据的候选不可复现，
不能作为最终版本接收。

## 6. 评价新因子

评价指标、公开因子库增量门槛和状态定义统一见
`docs/factor_research_plan.md` 第 5—6 节。队友不修改评价实现，只执行：

```bash
conda run --no-capture-output -n quant python -m pytest -q
```

取得经过核验的研究快照后，在本地正式运行 S：

```bash
PYTHONPATH=src conda run --no-capture-output -n quant \
  python scripts/run_first_round.py --resume-metrics
```

该入口会生成
`data/factors/candidate_pool.parquet`，列严格为
`date、instrument、candidate_id、factor_version、factor`。它是组合层的标准
自研候选输入。`--resume-metrics` 只能在已有候选公式和输入均未变化时使用，
脚本不会自动识别旧候选的代码变化；修改已有候选时必须同时使用
`--refresh-candidate CANDIDATE_ID`，新登记候选因旧报告中没有同名记录，会自动
正式计算。

## 7. 进入组合层

组合顺序、时间切分、模型和准入门槛统一见
`docs/factor_research_plan.md` 第 6—8 节。组合代码对队友只读；队友如需确认自己的
候选能进入动态接口，只运行：

```bash
PYTHONPATH=src conda run --no-capture-output -n quant \
  python scripts/run_combinations.py --check
```

`--check` 只验证合成数据合同，不产生有效性结论。统一组合训练、冻结和
`composite/` 登记由主仓库负责人完成。

主仓库负责人使用已核验快照正式运行 I、T 和三条组合管线：

```bash
PYTHONPATH=src conda run --no-capture-output -n quant \
  python scripts/run_combinations.py
```

I 与 T 的内容寻址缓存分别位于 `data/cache/incremental_v3/` 和
`data/cache/tree_v3/`，正常运行会自动复用完全相同的输入。新增或修改候选只会
生成包含该候选的新缓存键；只有完成候选级与联合池确认的成员才能写入冻结池。
不得通过删除缓存、改报告或沿用原编号绕过冻结验证。需要排查尚未冻结候选时，
使用 `--refresh-incremental-candidate CANDIDATE_ID` 或
`--refresh-tree-candidate CANDIDATE_ID` 精确重算；冻结候选发生机制或取值变化
必须登记为新版本并重新走完整准入。

## 8. 比赛提交与代码交付

候选完成本地真实快照评价并通过 AIStudio 短窗验收、结果值得提交时，同队成员及其 AI
可以直接上传比赛，不需要再次询问队长。

提交准入和冻结内容统一见 `docs/factor_research_plan.md` 第 9 节；Notebook
接口统一见 `docs/data_contract.md` 的“候选输出合同”。本文件只规定协作交付。

正式提交只能在比赛页面点击顶部“提交代码”，从队伍的 AIStudio 私有工作区选择
Notebook，再在“提交记录”确认新记录出现。以下入口均不属于比赛评测提交，禁止使用：

- AIStudio 的“分享策略”；
- “公开分享”或任何 `codeshare` 链接；
- 比赛“代码”标签页里的“上传代码”。

比赛提交只授权评测系统读取私有 Notebook，不代表授权向策略社区公开源码。

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
- 只修改本候选及“修改权限”允许的配套文件；
- 完整测试通过；
- 没有数据文件、密钥或 Notebook 输出；
- plan 规定的评价和冻结证据已经提供；
- 新数据依赖已按第 5 节交付；
- 最终代码已创建 MR/PR。
