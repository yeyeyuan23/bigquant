# Scripts 目录说明

这里放的是可直接执行的研究与交付工具，不是 `src/` 模块的一部分。运行命令时
默认当前目录是仓库根目录；大体量运行时数据仍统一放在 `/root/autodl-tmp/data`。

## 目录

| 位置 | 用途 |
|---|---|
| `runners/` | 跨多个 Python 步骤的长流程 Shell 入口 |
| `aistudio/` | 复制到 AIStudio 执行的导出、聚合与前缀探针 |
| `transfer/` | GitHub Release 数据传输与 SHA256 校验工具 |
| `factor_wiki_remaining/` | Candidate454 因子 Wiki 补全、血缘和前缀不变性检查 |
| `assets/` | 生成提交包时嵌入的静态辅助文件 |
| 顶层 `*.py` | 单一职责的构建、训练、评分、审计或提交工具 |

顶层 Python 脚本按动词识别职责：

| 前缀 | 职责 |
|---|---|
| `build_`, `prepare_`, `convert_`, `install_`, `upgrade_` | 数据或特征产物构建 |
| `evaluate_`, `train_`, `predict_`, `replay_` | 模型训练、推理与冻结模型复放 |
| `score_`, `diagnose_`, `finalize_`, `ensemble_`, `combine_`, `merge_` | 评分、诊断与路线合并 |
| `audit_`, `validate_`, `preflight_` | 准入、血缘和不变性检查 |
| `build_*_submission.py` | 提交包构建 |

部分历史训练脚本仍保留在 `scripts/` 顶层，因为冻结产物和测试记录了它们的精确
路径；只为美观移动会破坏可复现性。例如
`scripts/run_m_expanding_history_retrain.sh` 是正式 M 路线保留的训练来源。

`scripts/run_temporal_residual_stage.sh` 只作为历史记录保留：它依赖已退役的
`merge_strict_oos_routes.py` 和 `orthogonalize_oos_route.py`，在当前工作树中不能独立运行。

## 统一流水线入口

| 命令 | 作用 |
|---|---|
| `bash scripts/runners/run_factor_wiki_2019_2024.sh` | 构建 2019–2024 因子 Wiki 补全部分 |
| `bash scripts/runners/run_candidate454_assembly_after_components.sh` | 等待各组件并组装、验证 Candidate454 store |
| `bash scripts/runners/run_unified_after_candidate454.sh` | Candidate454 验证完成后启动统一实验 |
| `bash scripts/runners/run_unified_alpha_fusion_suite.sh` | 运行统一 Elastic Net、T、X、M 专家与 J 评分 |
| `bash scripts/runners/run_unified_full_experts.sh` | 先准备微观结构 store，再运行完整统一实验 |

新增长流程时放入 `runners/`；单步骤工具留在顶层并使用上表中的动词前缀。移动已有
脚本前，必须同步更新文档、生成器、测试和冻结产物中的路径引用。

传输命令同样从仓库根目录运行：

```bash
bash scripts/transfer/download_transfer_release.sh
bash scripts/transfer/upload_transfer_release.sh
```
