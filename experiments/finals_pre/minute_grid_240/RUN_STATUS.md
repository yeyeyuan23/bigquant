# 本地轮询交接

- 启动：2026-09-06 00:50:22（北京时间）。远程 `autodl-bigquant`，4090 D，16 CPU，80 GiB 内存。
- 远程目录：`/root/autodl-tmp/minute-grid-240-20260906`，原训练仓库和 submissions 未覆盖。
- 状态与日志：`results/status.json`、`results/clock240.log`、`results/clock242.log`、`results/runner.log`。
- 顺序：clock240 → clock242 → 行业/Barra 中性化评分。seed 20260801，各 3 epoch。1213 个训练日，241 个评价日。
- 当前对照：修正前 c0dcd219 的 fixed-clock 242 位；不是原 submission 的末尾补空布局。
- 已通过 37 项模型/加载测试，包括拒绝复用 242 位权重、两种 packer 在缺失分钟与午休边界的一致性。三天真实数据的值与 mask、初始化权重（217953 参数）、三轮训练顺序的校验和逐项一致，唯一差别为分钟网格。
- 首 50 步约 24.8 秒；初估全程 60–70 分钟，约 01:50–02:00 完成。按后续日志修正，不用历史耗时冒充测速。
- 本地 heartbeat automation ID：`240`。每 10 分钟检查，绑定当前任务，不另开云端任务。

本地仓库下执行：

```bash
python3 experiments/finals_pre/minute_grid_240/monitor.py --shutdown
```

运行中仅返回状态；结束（包括失败）后把远程全部结果复制到本地 `reports/dependencies/finals_pre/minute_grid_240/20260906/` 并核对 SHA-256，确认 GPU 没有其他进程后执行远程 shutdown。网络权限按工具要求申请。不要在本机执行 shutdown。

遇到失败先检查 runner.log 和当前 arm 日志；不盲目重复跑满整组，不覆盖已有 checkpoint。若只是评分脚本失败，可用已保存因子修复评分，无须重训。无可恢复任务时保存失败日志后关机。

收到关机执行返回后，再验证 SSH 是否已不可达；超时只能表述为机器离线，不能凭空声称已核验平台计费状态。成功收集并处理完毕后，将本 automation 更新为 PAUSED（保留其他配置），向用户报告两组结果和配对差值。新指标只能标为本次单种子实验，不能替换展示中的原官方成绩。

源码提交后，结果摘要与运行证据也应提交推送；大 checkpoint 和全量 parquet 保存在本地与机器数据盘，不强行入 Git。
