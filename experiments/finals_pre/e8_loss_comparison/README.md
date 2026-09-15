# E8：SmoothL1、L1、L2 辅助损失对照

状态：2026-09-15 16:56（北京时间）九次全新训练、评分全部完成，小型结果已收回并独立复算通过。运行记录见 [RUN_STATUS.md](RUN_STATUS.md)。

三种损失的 RankIC 接近，两项对比的 95% 区间均跨零，Holm p 均为 1.0。本轮没有证据说明 SmoothL1 优于 L1 或半平方 L2；这不等于三者效果等价。

本轮只回答：在当前 0.05 辅助系数及三轮训练预算下，替换辅助误差项是否改变 2024 历史验证期的 RankIC？不宣称各损失分别调参后的最优性能。

| 配置 | 每个交易日的训练目标 |
|---|---|
| smooth_l1 | 负 Pearson + 0.05 × mean SmoothL1，beta=1 |
| l1 | 负 Pearson + 0.05 × mean 绝对误差 |
| l2_half | 负 Pearson + 0.05 × 0.5 × mean 平方误差 |

L2 明确定义为半平方，使其与 SmoothL1 小误差段相同；相同系数不等于相同有效梯度强度。没有加入仅负 Pearson 的第四组，因此本轮不回答辅助项是否必要。

## 固定条件

沿用 E9 的实际运行源码及冻结依赖 `_runtime/`，新增目录独立保存。模型为三块 TCN、3/15/60 核、D=96、17 通道、240 分钟。2019–2023 训练，最后训练日 2023-12-28，保留一个交易日标签隔离；2024 共 241 个评分日。训练和评分均使用下一交易日 O2C 标签，训练目标作当日截面排名变换。

每组 AdamW，学习率 4e-4、weight decay 1e-4、dropout 0.1、三轮；每日最多 1,200 只股票。沿用 E9 的 float16 AMP、初始 scaler=1024、全模型梯度范数裁剪 1.0。种子 20260801、20260812、20260823。相同种子的模型初始化、日期顺序、股票抽样完全配对；不复用旧模型权重。三个配置的种子和数据顺序不会依赖之前配置的结果。

记录每轮相关性、辅助误差、预测均值/标准差、平均绝对误差，以及绝对误差超过 1 的比例。该比例说明 SmoothL1 的线性段实际被使用多少。记录完整梯度裁剪前范数，避免把输出梯度性质直接当成参数更新力度。

## 运行

在 AutoDL 项目根目录执行：

```bash
/root/autodl-tmp/conda-envs/quant/bin/python -m pytest -q experiments/finals_pre/e8_loss_comparison/test_e8.py
/root/autodl-tmp/conda-envs/quant/bin/python -u experiments/finals_pre/e8_loss_comparison/run.py \
  --data-root /root/autodl-tmp/data \
  --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
  --exposure /root/autodl-tmp/exposure_2024_full.parquet \
  --output /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e8_loss_comparison/20260915 \
  --device cuda
```

`--preflight-only` 只执行完整输入审计和真实数据短程验证。`--resume` 校验封存输入与源码后跳过已完成配置；中断的配置从头重新训练，已记录失败的配置必须先调查。程序拒绝直接覆盖既有产物。

## 判断与结果

主指标为行业与完整 Barra 中性化后的日均 RankIC。九组均使用相同的 241 个评分日、240,567 个有效股票日期样本。下表为三种子的均值 ± 样本标准差；多空 Sharpe 未扣成本，仅作因子诊断。

| 辅助损失 | RankIC | RankIC IR | 多空 Sharpe | 压力日 ICIR | 训练加推理分钟 |
|---|---:|---:|---:|---:|---:|
| SmoothL1 | 0.025809 ± 0.001366 | 0.5909 ± 0.0144 | 5.857 ± 0.122 | 0.3322 ± 0.0427 | 27.38 |
| L1 | 0.025731 ± 0.001282 | 0.5888 ± 0.0109 | 5.607 ± 0.113 | 0.3340 ± 0.0445 | 27.33 |
| 半平方 L2 | 0.025853 ± 0.001305 | 0.5958 ± 0.0143 | 5.792 ± 0.301 | 0.3403 ± 0.0330 | 27.45 |

| 相对 SmoothL1 | ΔRankIC | 95% 区间 | Holm p | RankIC 更高的种子 |
|---|---:|---:|---:|---:|
| L1 | −0.000078 | [−0.000308, +0.000165] | 1.0000 | 1/3 |
| 半平方 L2 | +0.000044 | [−0.000355, +0.000415] | 1.0000 | 3/3 |

半平方 L2 在三个种子上的 RankIC 都略高，但配对差异不足以支持可靠改善。SmoothL1 的平均多空 Sharpe 较高，这一辅助指标没有做显著性检验，也不是扣费后的策略收益。

对固定三个种子的每日配对差值取平均，做同步 10 日移动块 bootstrap 10,000 次，报告 95% 区间；两项相对 SmoothL1 的双侧比较使用 Holm 校正。区间不覆盖全部初始化不确定性。2024 已参与研究，称历史验证期；不称全新未见测试。差异不显著不等于效果等价。

本次实测：RTX 4090 D，217,953 个参数，峰值 CUDA 已分配显存约 1.81 GiB。每次训练加推理平均 27.33–27.45 分钟；含准备和评分，总计 4 小时 16 分 24 秒。正式提交权重未替换。

## 结果与本地核验

- [汇总、配对差值与逐日数据](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/results/README.md)
- [本地独立复算记录](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/local_validation.json)
- [收回文件与源码校验和](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/collection.json)

本地已核验 69 个小型文件、23 份运行源码，按逐日指标重算四项汇总、种子标准差、配对差值、10,000 次块 bootstrap 与 Holm 校正，并检查九次训练的初始化、抽样、配置与评分日期一致性。浮点 32 位收益保存为 CSV 后，用 64 位重算 Sharpe 的最大差异为 8.8e-7。

```bash
python experiments/finals_pre/e8_loss_comparison/validate_results.py \
  reports/dependencies/finals_pre/e8_loss_comparison/20260915
```

本地复算从已保存的逐日指标开始；没有重新训练或重算股票层面的中性化。完整 checkpoint、因子与原始数据保留在 AutoDL，仓库保留其原始校验和及远端审计。源码与运行时快照保持实际运行版本。

发布检查：E8 专项 10 项、E9 专项 29 项及全仓 693 项通过（2 项因外部数据条件跳过）；Ruff 通过，源码覆盖率超过 75% 门槛。E8 和 E9 各自在独立 CI 进程运行，避免冻结依赖影响主仓库测试。记录见 [publication_validation.json](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/publication_validation.json)。
