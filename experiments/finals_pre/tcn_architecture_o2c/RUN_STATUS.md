# E9：TCN 结构实验运行状态

**45/45 已完成，全部评分与独立统计复核通过；小型结果已回收并校验，本轮结束时 AutoDL 已关机，定时检查已暂停。**

完整实验目的、每个配置的结论和成本见 [README.md](README.md)。正式提交模型、权重和历史成绩未替换。

- 15 个配置、3 个固定种子，每次三轮，全部重新训练；1,213 个训练日期、241 个有效评分日期、240,567 个评分键。
- 整轮训练、评分与聚合约 23 小时 46 分钟。
- 基线平均训练阶段 29.86 分钟，含推理总耗时 31.45 分钟；五分支分别 31.81、33.43 分钟，总耗时增加 6.3%。
- 没有候选通过主指标改善标准；只有原核宽四块的下降通过 Holm 校正。三块必要性、三分支必要性及 3/15/60 最优性均未得到证明。
- 29 项专项测试、45 次真实输入前向/反向预检查通过；完整运行后的源码、数据、实际样本、初始化、checkpoint、因子和逐日指标均已核验，bootstrap 和 Holm 已独立复算。

## 产物与关机记录

- [最终汇总](../../../reports/dependencies/finals_pre/tcn_architecture_o2c/results/summary.csv)
- [完整时间表](../../../reports/dependencies/finals_pre/tcn_architecture_o2c/results/timing_summary.csv)
- [独立完成审计](../../../reports/dependencies/finals_pre/tcn_architecture_o2c/audits/completion_audit.json)
- [本地结果回收校验](../../../reports/dependencies/finals_pre/tcn_architecture_o2c/handoff/local_receipt.json)：包含全部 45 份逐日指标；结果原始文件仍可用清单校验。
- [关机凭据](../../../reports/dependencies/finals_pre/tcn_architecture_o2c/handoff/shutdown_receipt.json)：执行关机命令返回 0，AutoDL 实例列表显示“已关机”。

代码直接维护于 AutoDL：/root/autodl-tmp/projects/bigquant-default/experiments/finals_pre/tcn_architecture_o2c/。checkpoint、完整因子、原始分钟数据仍留在远端；本地只保留 Markdown、小型汇总、manifest 和逐日指标，不恢复实验 Python 代码。新增文件不用日期命名。

此前阶段审计和 timing_progress 作为阶段记录保留；完整结论及成本以上述最终结果为准，不再将 30/45 或 38/45 当作当前状态。预检查中的初始 AMP 缩放修正、输入合同和完整固定协议见 README。
