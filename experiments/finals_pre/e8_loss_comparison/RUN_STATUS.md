# E8 运行状态

2026-09-15：九次训练及评分全部完成，结果已收回，本地独立复算通过。

- 开始：2026-09-15 12:39:35（北京时间）。
- 完成：2026-09-15 16:56:00（北京时间）。
- 总耗时：15,384 秒，约 4 小时 16 分 24 秒。
- 完成状态：`state=complete`、`completed=9`、`total=9`，结果审计记录为九组。原始状态中的 `phase=training` 是最后一组留下的阶段字段，不是当前运行状态。
- 配置：三种辅助损失 × 三个种子，每次三轮、全新权重。结果及解释边界见 [README.md](README.md)。
- 硬件：RTX 4090 D 24 GB；PyTorch `2.6.0+cu124`，Python 3.11.15。
- 单组训练加推理：平均 27.33–27.45 分钟。
- 代码：`/root/autodl-tmp/projects/bigquant-default/experiments/finals_pre/e8_loss_comparison/`。
- 远端结果：`/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e8_loss_comparison/20260915/`。
- 本地小型结果：[20260915](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/results/README.md)，69 个文件共约 1.31 MB；完整数据、checkpoint 和因子未下载。
- [复算记录](../../../reports/dependencies/finals_pre/e8_loss_comparison/20260915/local_validation.json)：文件与运行源码校验和、三种子配对、逐日指标、汇总、配对差值及统计推断通过。
- 正式提交模型未替换。

此前曾设置 18:15 自动关机，但本次收回结果时 SSH 仍可连接，不能据此记录为“已经关机”。本次仅收回、核验和发布已完成的实验。
