# Reports 目录地图和字段字典

`reports/` 是本地评价和提交路线审计目录。报告是审计产物，不是冻结状态；
正式冻结成员优先看 `latest/factor_pool_decisions.json` 和对应
`data/cache/*/frozen_state.json`。平台真实分数仍以提交结果为准。

## 目录层级

```text
reports/
├── README.md
├── latest/                         # 当前组合运行摘要入口
│   ├── factor_pool_check.json      # 当前真实快照合同
│   ├── combination_summary.csv     # 当前三条最终路线排序
│   ├── factor_pool_decisions.json  # 当前机器可读路线合同
│   └── factor_pool_admission.csv   # 当前候选进入 S/I/T 的总表
├── first_round/                    # 候选第一轮单因子诊断
├── routes/                         # S/I/T 和最终路线细项
├── diagnostics/                    # 一次性研究、探针和覆盖率附件
└── archive/                        # 旧流程语义或历史残留
```

## 推荐阅读顺序

1. `latest/factor_pool_check.json`：确认数据快照、候选版本、覆盖率和 join 规则。
2. `latest/combination_summary.csv`：看当前本地 J proxy 排序。
3. `latest/factor_pool_decisions.json`：看每条 pipeline 的正式成员和完整评分合同。
4. `latest/factor_pool_admission.csv`：看每个自研候选进入 S/I/T 哪些路线。
5. `routes/`：需要排查具体 S/I/T 准入时再看。
6. `first_round/`：需要看候选单因子技术、IC、稳定性、相关性时再看。
7. `diagnostics/` 和 `archive/`：只用于追溯。

## latest/ 当前入口

### factor_pool_check.json

- `status`：真实快照合同检查结果，`ok` 才能继续正式训练。
- `rows`：组合主面板的股票日行数。
- `duplicate_keys`：`date, instrument` 重复键数量，必须为 0。
- `factorlib_reference.screened_features`：冻结 screened15。
- `factorlib_reference.j_reference_columns_present`：本地 J reference 可用列数量，
  当前应等于 all36 公开列数加 `INCLUDE_IN_J_BASELINE = True` 的候选列数。
- `candidate_pool_reference.factor_version`：当前候选长表版本。
- `candidate_pool_reference.sha256`：当前候选长表文件哈希。
- `combination_inputs.all_candidate_count`：进入组合层的自研候选数量。
- `minimum_public_coverage`、`minimum_self_coverage`：公开因子和自研因子的最低覆盖率。

### combination_summary.csv

三条最终路线的验证期表现、稳健 J 和冻结排序。

- `pipeline`：`self_factor_composite`、`joint_elastic_net` 或 `joint_lightgbm`。
- `method`：组合方法。
- `selected_direction`：本地 J proxy 选择的方向。
- `validation_*_score_proxy`：本地 J proxy 的分场景分数。
- `robust_score_proxy`：路线排序用的稳健分数。
- `rank`：本地冻结提交顺序。

### factor_pool_decisions.json

机器可读的完整合同，包括 S/I/T 协议、J 评分协议、最终路线成员和排序。

优先看：

- `frozen_winner`
- `frozen_submission_order`
- `pipelines.self_factor_composite.features`
- `pipelines.joint_elastic_net.features`
- `pipelines.joint_lightgbm.features`

### factor_pool_admission.csv

每行对应一个自研候选的最终路线归属。

- `candidate_id`：无 `self__` 前缀的候选编号。
- `feature`：带 `self__` 前缀的候选特征名。
- `single_factor_passed`：是否进入 S 路线。
- `individual_I_passed`：是否在通过 S 后继续通过 I entry gate。
- `elastic_net_pool_input`：兼容旧报告的别名，等同于 `individual_I_passed`。
- `incremental_evaluation_status`：I 路线最终状态。
- `enters_self_factor_composite`：是否进入最终规则复合。
- `enters_joint_elastic_net`：是否进入最终 Elastic Net。
- `tree_incremental_passed`：是否进入最终 LightGBM。
- `enters_joint_lightgbm`：是否进入最终 LightGBM。
- `route_count`：该候选进入的路线数量。
- `status`：`admitted` 或 `rejected`。

## first_round/

这些文件用于候选单因子诊断，不等同于最终 S/I/T 冻结成员。

- `first_round_technical.csv`：覆盖率、活跃日、缺失和重复等技术检查。
- `first_round_metrics.csv`：单因子的 Rank IC、t 统计和分期诊断。
- `first_round_stability.csv`：跨期稳定性。
- `first_round_correlations.csv`：候选之间的相关性。
- `first_round_decisions.json`：第一轮机器可读诊断结果。严格漏斗正式运行会重新对
  全候选执行 S；该文件不能预先否决候选，也不能替代 frozen S。

## routes/

### S 路线

- `single_factor_route_admission.csv`：S 路线的候选级路线审计。当前 S 满足
  质量、稳定性和低重复要求后直接进入 family-balanced 规则复合。
- `single_factor_route_promotion.csv`
  - `passed`：S 路线是否通过质量、稳定性和低重复要求。
  - `reasons`：失败原因。
  - `retained_pending_candidates`：本次满足 S 要求的候选。

S 是否保留会读取 `s_*` 字段：覆盖率、活跃日、行业与 Barra 中性化后的 Rank IC、
最差 fold、正向 fold 比例、方向一致性、行业内有效性和相对当前 S baseline 的
最大 Rank 相关性。通过 trial 即进入 frozen S；准入阶段不计算 J。其他可交易和
分组诊断不再无条件增加主流程计算量；如果只是要审计更多单因子质量，用
`scripts/run_first_round.py`，如果要给最终路线补审计字段，再给
`scripts/run_combinations.py` 加 `--include-route-diagnostics`。

### I 路线

- `factor_pool_incremental.csv`：I 路线候选 entry/诊断审计。
- `incremental_pool_promotion.csv`：I entry 通过池的最终冻结记录。
- `incremental_factorwise_admission.csv`：I 的逐候选 entry 诊断和准入状态。
- `incremental_factorwise_promotion.csv`：I entry 通过池的整体记录。
- `joint_elastic_net_weights.csv`：最终 Elastic Net 权重。
- `joint_elastic_net_metrics.csv`：最终 Elastic Net 路线指标。

常用字段：

- `candidate`：带 `self__` 前缀的候选特征名。
- `i_trial_passed`：是否通过 I entry gate。
- `i_rank_ic_mean`：候选原始日度 Rank IC 均值。
- `i_residual_rank_ic`：候选对 `screened15 + frozen_I` 日内秩残差化后的 Rank IC。
- `i_max_abs_rank_correlation`：候选相对 `screened15 + frozen_I` 的最大绝对秩相关。
- `active_days`：开发期内该候选有截面变化的交易日数量。
- `individual_passed`：I entry gate 是否通过。
- `frozen_after_validation`：是否进入 frozen I。
- `evaluation_status`：当前候选在 I 路线的最终状态。

### T 路线

- `tree_factor_admission.csv`：T 路线候选准入状态，组合流程可用作旧状态输入。
- `tree_factor_incremental.csv`：T 路线正交 entry/诊断审计。
- `tree_factorwise_admission.csv`：T 的逐候选正交 entry 诊断和准入状态。
- `tree_factorwise_importance.csv`：最终联合 LightGBM 训练的 split/gain importance。
- `tree_factorwise_promotion.csv`：T 正交通过池的整体记录。
- `joint_lightgbm_metrics.csv`：最终 LightGBM 路线指标。

常用字段：

- `tree_data_eligible`：是否满足 T 的开发期有效日门槛。
- `individual_passed`：T 正交 entry gate 是否通过。
- `tree_incremental_passed`：是否进入 frozen T。
- `evaluation_status`：当前候选在 T 路线的最终状态。

`tree_factorwise_importance.csv` 额外字段：

- `candidate`：触发本次训练的候选；轻量 T 的最终联合模型使用 `__joint_lightgbm__`。
- `evaluation_stage`：轻量 T 下为 `final_joint`。
- `baseline_candidates`：轻量 T 下为最终联合模型包含的自研 T 候选。
- `train_start`、`train_end`：训练窗口。
- `test_start`、`test_end`：样本外预测窗口。
- `feature`：LightGBM 输入特征名。
- `split_importance`、`gain_importance`：LightGBM importance。

### 参考和最终路线

- `factor_pool_screening.csv`：公开 screened15 和候选筛查结果。
- `competition_J_reference_directions.csv`：本地 all36 J proxy 方向参考。
- `self_factor_composite_metrics.csv`：S winner 路线指标。

`self_factor_composite_metrics.csv`、`joint_elastic_net_metrics.csv` 和
`joint_lightgbm_metrics.csv` 默认只保证包含路线级 J 排名需要的字段。
`validation_*_rank_ic_*`、`*_tradable_*` 等字段只有在
`--include-route-diagnostics` 下才会计算；默认可能是空值，不能拿来做
准入判断。

## diagnostics/

一次性研究、平台探针和数据覆盖率附件，不是稳定入口。

- `aistudio_lightgbm_lookahead_probe_2026-07-27.json`
- `market_state_coverage.json`
- `oap_signal_screening.md`
- `oap_signal_shortlist.csv`

## archive/

旧流程语义或容易误导的报告放在 `archive/`。目前：

- `archive/legacy_pre_factorwise_20260728/`
  - `incremental_direct_pool.csv`
  - `incremental_forward_admission.csv`
  - `tree_pool_promotion.csv`

这些文件对应旧的 direct-pool / pre-factorwise 语义，不应再作为当前准入标准阅读。
