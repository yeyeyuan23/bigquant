# Reports

本目录是历史研究汇总。行情、暴露、标签、逐日序列、训练权重与 checkpoint 已从公开版本移除。保留的 manifest、审计状态和哈希描述历史运行，不能证明当前公开文件齐全；复现需自行提供有权使用的产物。

`reports/` 只保留三个入口，避免把当前评分、复算依赖和历史实验混在一起。

```text
reports/
├── README.md
├── latest/          # 当前有效的 2019-2024 六年 J 结果
├── dependencies/    # 历史复算协议、训练汇总与审计说明（不含预测/权重）
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

保留各路线的协议、历史指标与审计说明。目录名沿用原研究布局；同名 checkpoint、模型文本、预测 Parquet 和逐日 CSV 并未随本版本提供。推理入口与所需模型格式见 `submissions/ARTIFACT_CONTRACT.md`。

## archive

`archive/legacy_factor_pool/` 保存旧 S/I/T 因子池报告和旧说明，
`archive/legacy_pre_factorwise/` 保存更早的准入流程结果。它们不属于当前
六年 J 排名，不应与 `latest/` 混用。

新的实验中间输出放在 `work/`；只有选定并通过校验的结果才进入 `reports/latest/`。
