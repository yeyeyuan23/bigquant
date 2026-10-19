# E6 历史结果

协议、训练代码位置与结论见 [E6 说明](../../../../../experiments/finals_pre/e6_channel_groups/README.md)。本目录是历史结果补档；当前 240 分钟入口的新结果使用独立的 `clock240/` 子目录。

- [a4_per_seed.csv](a4_per_seed.csv)：五个配置 × 三个种子，共 15 行。
- [a4_summary.csv](a4_summary.csv)：四项指标的三种子均值和样本标准差。
- [a4_paired_deltas.csv](a4_paired_deltas.csv)：完整模型减去删减版，正数表示完整模型较高；`full_higher_runs` 为三种子中完整模型较高的次数。
- [audit.json](audit.json)：原始标签、评价期、42 个中性化回归项和通道分组。
- [publication_audit.json](publication_audit.json)：PRE 来源、GitHub 文件对象和 SHA256，以及本次补档的汇总复算范围。

原始四文件与已发布的 PRE 副本逐字节一致。本地核验从逐种子指标开始，没有重新训练、重算股票层面中性化或逐日收益；不声称已恢复完整的远端运行产物。
