# Reports

`reports/` 只保留三个入口，避免把当前评分、复算依赖和历史实验混在一起。

```text
reports/
├── README.md
├── latest/          # 当前有效的 2019-2024 六年 J 结果
├── dependencies/    # 重放、复算和提交所需的冻结预测、checkpoint 与训练审计
└── archive/         # 已停止使用的旧报告和旧评分流程，仅供追溯
```

## 当前结果

当前正式比较表：

- `latest/J_2019_2024_18_ROUTES_FULL.csv`
- `latest/J_2019_2024_18_ROUTES_DETAILED.json`
- `latest/J_2019_2024_18_ROUTES_MANIFEST.json`
- `latest/J_2019_2024_18_ROUTES_VALIDATION.json`

主表覆盖 2019-2024 连续六年、18 条路线。固定展示列依次为
`group, route, family, years, J, A, B, score_proxy, a_proxy, b_proxy`、A 四项原值、
A 四项分位数、B ModelScore/mean/std/nonzero 和 `score`；其中 `score` 是 J 的稳定
展示别名。共同评分天数、窗口数、Candidate454 参考因子数、联合路线数和公共行数
作为运行审计列排在其后。

`J = 0.3 * A + 0.7 * B`，没有 `J_stable`。该结果包含 checkpoint 的训练期，
因此不是 strict OOS，也不是平台官方分。

## dependencies

这里不是“旧结果”，而是当前结果可复核所需的依赖，不能随意删除：

- `candidate454_joint_full_history_2019_2024_20260804/`：基础 10 路 route batch 与评分审计。
- `frozen_full_history_replay_2019_2024/`：X/M/T 的六年冻结权重纯推理结果。
- `m_expanding_history_retrain_20260803/`、`m_v3_final_2019_2024_seed_20260803/`：M checkpoint。
- `m_multiaxis_full_20260804/`：三轴 M 的代码对应 checkpoint、日志和审计；明确保留。
- `temporal_incremental_objective_20260803/`：EN454 与 residual-T 依赖。
- `unified_alpha_fusion_suite_20260801_full_experts_v1/`：X-tree checkpoint。
- `xt_protocol_retrain_20260803/`：X-MLP/T checkpoint。

## archive

`archive/legacy_factor_pool/` 保存旧 S/I/T 因子池报告和旧说明，
`archive/legacy_pre_factorwise_20260728/` 保存更早的准入流程结果。它们不属于当前
六年 J 排名，不应与 `latest/` 混用。

新的实验中间输出放在 `work/`；只有选定并通过校验的结果才进入 `reports/latest/`。
