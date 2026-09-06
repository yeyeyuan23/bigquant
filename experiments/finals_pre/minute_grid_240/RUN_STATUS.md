# 补充两个 seed：运行与本地轮询交接

用户已授权补跑 20260812、20260823。每个 seed 均为 clock240 → clock242，各 3 epoch，共四次训练。完成后与已有 seed 20260801 汇总均值、样本标准差、配对差值及同向次数；不要根据单个 seed 选择性报告。

2026-09-06 13:21:00（北京时间）已启动，总进程 PID 1511。首 50 步 24.84 秒；预计 15:20–15:35 完成。已确认原训练源码和数据校验和匹配。

- 训练机：`autodl-bigquant`，4090 D，已确认在线且启动前空闲；数据盘剩余 7.6 GiB。
- 沿用首次实验代码快照：`/root/autodl-tmp/minute-grid-240-20260906/clock240` 和 `clock242`。模型、packer、训练器以及 store/标签/Barra 暴露校验和必须与第一次配对一致。
- 新驱动：`/root/autodl-tmp/minute-grid-240-moreseeds-20260906/code/run_two_more.py`。
- **新总状态**：`/root/autodl-tmp/minute-grid-240-moreseeds-20260906/results/status.json`。以这个总状态决定结束，不能用旧实验的 completed 状态或某个 seed 的完成状态关机。
- 新日志：总目录的 `runner.log`；`seed20260812/clock240.log`、`clock242.log`；`seed20260823/clock240.log`、`clock242.log`。
- 原 seed 用时 58 分 55 秒，两个新增 seed 预计约 2 小时，按日志实际速度修正。
- 本地 heartbeat automation ID `240`，继续每 10 分钟检查，prompt 简短；只报告完成、异常或需要处理的情况。

在本地研究仓库执行以下命令：

```bash
python3 experiments/finals_pre/minute_grid_240/monitor.py --shutdown \
  --remote-root /root/autodl-tmp/minute-grid-240-moreseeds-20260906/results \
  --local-root reports/dependencies/finals_pre/minute_grid_240/additional_seeds_20260906
```

该命令只在两个 seed 的总任务结束后拉回结果、校验 SHA-256，并在确认 GPU 无其他进程后执行远程 shutdown。后续再次验证 SSH 状态，并停用本 automation。不要使用不带新路径参数的旧命令。

若总状态 failed，先读 runner.log 和当前 seed 日志。已有完整 checkpoint/因子时优先修复评分，无需重训。禁止覆盖第一次配对产物。

完成后核对 `three_seed_summary.csv/json`、`three_seed_per_run.csv`、`three_seed_paired_deltas.csv`，并从逐日 IC 复核均值/ICIR。结果摘要、日志及审计证据应提交推送；checkpoint、全量 parquet 与源码快照留在本地及远程数据盘。原 seed 结果仍位于 `reports/dependencies/finals_pre/minute_grid_240/20260906/`。

这里的 242 基线是修改前 src 固定时钟布局，不能改称原 submission 尾部补空布局。三种子覆盖相同 2024 年，不代表增加了独立样本外时期。
