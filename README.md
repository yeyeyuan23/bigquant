# BigAlpha 2026 — Unified Alpha（M_raw 冠军线）

用原始分钟盘口和 Candidate454 因子工程构建互补的 OOS 专家，最终以 M 专家
（原始微观结构）单因子作为正式提交并进入决赛。平台正式成绩（提交
`6d0fe02c-1b61-41bd-9de9-fe9ac0774ffd`，checkpoint sha256 `252c39ba…`）：

| 口径 | 总分 | A | B |
|---|---:|---:|---:|
| 公榜 | **0.95480** | 0.95106 | 0.95641 |
| 私榜 merge | **0.88736** | 0.95788 | 0.85714 |

`总分 = 0.3 × A + 0.7 × B`。逐时段读数与答辩材料以 Mac 端
`BigAlpha2026_Pre/final_platform_scores.md` 为权威（本仓库为私有仓库）。

## 我们干了什么

1. **候选因子工程（7 月，已退役）**：登记-实现-准入流水线（S/I/T）筛出
   PV/HF/OB/FR/composite 共 454 个候选，冻结为 Candidate454 因子面板。
   S/I/T 评价与组合管线代码已从工作树移除，历史见 `origin/main` 与归档 tag。
2. **统一三专家（8 月上旬）**：
   - `T / Factor Temporal`：Candidate454 × 60 日，CNN + Transformer + DeepSets；
   - `X / Factor Cross-sectional`：当日 Candidate454，LightGBM 主模型 + MLP 挑战；
   - `M / Raw Microstructure`：17 通道分钟面板（前三档盘口，午休隔离、缺失保留），
     多尺度因果 TCN + 显式统计双通路，DeepSets 横截面上下文；
   - `Baseline`：Candidate454 全池滚动 Elastic Net（EN454）。
   本地统一用 J 代理（`competition_score_proxy`）比较路线。
3. **提交演进**：EN454 → X-tree → T/M 残差链 → **M_raw expanding-history e3**
   （2019→2024 扩窗训练，最终 checkpoint 见
   `reports/dependencies/m_expanding_history_retrain_20260803/final_full_history_e3/`）。
   高相关变体清库后只保留低相关组合，B 项从中游回升至 0.956。
4. **决赛复盘实验（8 月 19 日起，进行中）**：`experiments/finals_pre_20260819/`，
   E0 重建 holdout、E2/E2b/E2c 卷积核与 seed 消融、E4 多折 walk-forward、
   E7 N-score 增量贡献体系，为答辩提供证据页。见该目录 README。

## 仓库布局

```text
src/bigalpha2026/
├── alpha_models/          # T/X/M 网络、数据合同、训练数据装配
├── candidates/            # 454 候选的因子实现（pv/hf/ob/fr/composite）
├── candidate_transforms.py
├── feature_contracts.py
├── factor_pool.py / factorlib.py / evaluation.py / research_policy.py
│                          # J 参考池层：B 项代理的公开库筛选、方向冻结与策略窗口
└── competition_score_proxy.py   # 本地 J 评分器（自包含日频指标原语）
scripts/                   # 数据装配、专家训练评估、提交构建、审计
scripts/j_reference_inputs.py    # J 评分的参考输入装载（从退役管线中抽取的现役闭包）
experiments/finals_pre_20260819/   # 决赛实验 lane（见其 README）
submissions/               # 冻结提交包（见其 README；冠军 = m_expanding_history_e3_20260803）
reports/{latest,dependencies,archive}/   # 见 reports/README.md
artifacts/                 # 冻结与传输清单
```

## 数据与环境（隔离约定）

代码树内不放大体量数据；所有 store 位于 `/root/autodl-tmp` 下的独立路径：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/data/` | 运行时数据合同：`labels/`、`universe/`、`exposures/`、`factors/`（candidate_pool）、`features/`（FACTORLIB / ALL36 / MICRO_DAILY_FULL 等）、pool manifest |
| `/root/autodl-tmp/data/e2e_maps/` | e2e instrument 映射表与审计 report |
| `/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/` | M 训练用 17 通道分钟 store（schema v2，1456 交易日） |
| `/root/autodl-tmp/candidate454_completion_full_2019_2024/candidate454_store/` | Candidate454 特征 store |
| `/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate454_store/` | 462 补全版 store（E7 使用） |
| `/root/autodl-tmp/m_v3_deep_book_context_2019_2024/` | L5 日级 deep-book context（v2/L5 挑战线用） |
| `/root/autodl-tmp/m_e2e_raw_store/` | e2e 压缩原始分钟数据 |
| `/root/autodl-tmp/conda-envs/quant/bin/python` | 运行环境（torch 2.6.0+cu124） |

`data/` 目录只保留 manifest 与小文件。老 SITJ 数据与旧 worktree 已删除；
如需重建 store，原始数据以平台/AIStudio 为准。

## 代码入口

```text
src/bigalpha2026/alpha_models/{temporal,tabular,microstructure,microstructure_v2}.py
scripts/prepare_unified_microstructure_store.py     # 分钟 store 构建（e2e_compressed / canonical 两 profile）
scripts/evaluate_unified_{temporal,mlp,tree,elasticnet,microstructure}.py
scripts/train_unified_final_checkpoint.py
scripts/build_unified_m_raw_submission.py           # 冠军提交包构建
scripts/score_submission_j_stability.py             # J 稳定性评分
scripts/replay_frozen_candidate454_models.py
```

M 训练器只接受审计版 schema v2 store；profile 规则（e2e 价格/成交额 ÷100、
instrument 一一映射，canonical 不转换）见 `docs/unified_alpha_plan.md`。

## 验证

```bash
python -m ruff check .
coverage run -m pytest -q && coverage combine && coverage report --include="src/*"
```

两者当前均为绿色。本地/AutoDL OOS、AIStudio 真实执行、前缀一致性、提交状态与
平台分数是不同证据层，只有完成对应验证后才能声明该层通过。

## 历史与归档

- SITJ 时代（候选准入、公开因子库、三组合管线）：`origin/main`、
  tag `archive/main-sitj-20260731`；工作树中不再保留。
- M multiaxis / v2 / v3 挑战线：结果与冻结产物在 `reports/` 与 `submissions/`，
  实验分支见 tag `archive/feat-m-multiaxis-full-20260804`。
- 决赛演示与平台分数记录在 Mac 端 `~/Projects/bigquant/BigAlpha2026_Pre/`。
