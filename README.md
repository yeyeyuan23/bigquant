# BigAlpha 2026 Unified Alpha

当前目标是用 Candidate454 因子工程和原始分钟盘口构建三个彼此互补的 OOS 专家，
并相对 Candidate454 全池 Elastic Net 基线最大化稳定 Competition Score。

唯一计划文档：[Unified Alpha Plan](docs/unified_alpha_plan.md)。

## 当前架构

- `T / Factor Temporal`：Candidate454 的 60 日 CNN + Transformer + DeepSets。
- `X / Factor Cross-sectional`：当日 Candidate454 的 LightGBM 主模型与 MLP 挑战模型。
- `M / Raw Microstructure`：原始分钟价格、成交和前三档盘口的 TCN/CNN + 显式统计双通路专家。
- `Baseline`：Candidate454 全池、60 日训练/20 日预测 Elastic Net OOS 路线。

M 的代码与严格 OOS 入口已经实现，但分钟数据尚未准备，当前没有真实训练结果。
统一套件只有在显式设置 `UNIFIED_MICROSTRUCTURE_STORE` 且 manifest 存在时才会启动 M；
否则写入 `microstructure_status.json` 并明确跳过，不会制造占位分数。

本地 E2E 压缩分钟数据可用下面的显式 profile 建 store：

```bash
python scripts/prepare_unified_microstructure_store.py \
  --source-profile e2e_compressed \
  --instrument-map data/e2e_parquet/instrument_id_map_internal_2019_2024.csv \
  --input data/e2e_parquet/bigalpha_2026_e2e_bar1m/*.parquet \
  --output-dir data/runtime/unified_microstructure_store
```

该 profile 固定执行价格与成交额 `/ 100`，并要求映射完整且一一对应。平台已是正式股票代码
和标准单位的数据使用 `--source-profile canonical`，不能同时传 instrument map。每个输入文件、
映射表和质量审计都会写入 manifest；训练器只接受审计版 schema v2。

不存在 Bar156 或 All618。容量版本只是同一专家内的实验，不能自动视为独立专家或
默认等权进入最终融合。

## 代码入口

```text
src/bigalpha2026/alpha_models/temporal.py
src/bigalpha2026/alpha_models/tabular.py
src/bigalpha2026/alpha_models/microstructure.py
src/bigalpha2026/alpha_models/microstructure_data.py
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
