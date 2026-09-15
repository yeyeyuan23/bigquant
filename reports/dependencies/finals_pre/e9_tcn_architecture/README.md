# E9：TCN 架构实验结果

15 个配置 × 3 个种子，共 45 次重新训练；每次三轮，240 个固定分钟位置、17 通道。2019–2023 训练，2024 年 241 日历史验证期。

完整问题、每个配置的结论和复核方法见 [E9 实验说明](../../../../experiments/finals_pre/e9_tcn_architecture/README.md)。

- [汇总表](results/summary.csv)、[结构化汇总](results/summary.json)、[45 次原表](results/per_run.csv)、[逐种子配对差值](results/paired_deltas.csv)。
- 四组结果：[深度](results/depth.csv)、[分支数](results/branches.csv)、[核宽](results/kernels.csv)、[解释对照](results/controls.csv)。
- [训练时间汇总](results/timing_summary.csv)、[逐次计时](results/timing_per_run.csv)。
- 每次逐日指标保存在 runs/seed1、seed2、seed3 下的配置目录中；[每日 RankIC 立方体](results/daily_rankic_cube.npz)供复算。
- [完成审计](audits/completion_audit.json)、[封存配置及源码/数据哈希](prepared/prepared.json)、[原始结果哈希](results/audit.json)、[发布清单](publication_manifest.json)。

没有候选通过预先约定的主指标改善标准。原核宽四块的下降通过 Holm 校正；五分支三个种子的主指标均较低，总耗时增加 6.3%，但下降未通过校正。三块、三分支和 3/15/60 的必要性或最优性均未得到证明。

训练权重、完整因子、原始分钟数据与大体量抽样文件留在 AutoDL，未提交 Git。results/ 内原始汇总与审计均保持原始字节，不用发布文字改写覆盖历史记录。
