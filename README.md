# BigAlpha 2026 Unified Alpha

当前目标是用 Candidate462 因子工程和原始分钟盘口构建三个彼此互补的 OOS 专家，
并相对 Candidate462 全池 Elastic Net 基线最大化稳定 Competition Score。

唯一计划文档：[Unified Alpha Plan](docs/unified_alpha_plan.md)。

## 当前架构

- `T / Factor Temporal`：Candidate462 的 60 日 CNN + Transformer + DeepSets。
- `X / Factor Cross-sectional`：当日 Candidate462 的 LightGBM 主模型与 MLP 挑战模型。
- `M / Raw Microstructure`：原始分钟价格、成交和前三档盘口的 TCN/CNN + 显式统计双通路专家。
- `Baseline`：Candidate462 全池、60 日训练/20 日预测 Elastic Net OOS 路线。

M 的代码与严格 OOS 入口已经实现，但分钟数据尚未准备，当前没有真实训练结果。
统一套件只有在显式设置 `UNIFIED_MICROSTRUCTURE_STORE` 且 manifest 存在时才会启动 M；
否则写入 `microstructure_status.json` 并明确跳过，不会制造占位分数。

不存在 Bar156 或 All618。容量版本只是同一专家内的实验，不能自动视为独立专家或
默认等权进入最终融合。

## 代码入口

```text
src/bigalpha2026/alpha_models/temporal.py
src/bigalpha2026/alpha_models/tabular.py
src/bigalpha2026/alpha_models/microstructure.py
scripts/evaluate_unified_temporal.py
scripts/evaluate_unified_mlp.py
scripts/evaluate_unified_tree.py
scripts/prepare_unified_microstructure_store.py
scripts/evaluate_unified_microstructure.py
run_unified_alpha_fusion_suite.sh
```

候选因子实现仍保存在 `src/bigalpha2026/candidates/`，历史实验、冻结产物和报告保留在
Git 历史及各自目录中，但不再充当当前架构说明。

## 验证

```bash
python -m ruff check .
coverage run -m pytest -q
coverage combine
coverage report --include="src/*" --fail-under=75
```

本地/AutoDL OOS、AIStudio 真实执行、前缀一致性、提交状态和平台分数是不同证据层。
只有完成对应验证后才能声明该层通过。
