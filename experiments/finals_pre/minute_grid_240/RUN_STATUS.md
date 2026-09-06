# 240 位三种子实验：已完成并关机

2026-09-06（北京时间）新增 seed 20260812、20260823 已全部完成。每个 seed 均为 clock240 → clock242，各 3 epoch；与原 seed 20260801 合并为三个配对、六次训练。本任务已结案，不再执行轮询或关机命令。

- 13:21:00 启动，15:31:49 完成，新增四次训练及评分合计 2 小时 10 分 49 秒。
- 15:37:13 拉回并校验全部 47 个远程文件，15:37:14 执行 `autodl-bigquant` 远程 shutdown，命令返回 0；随后 SSH 连接关闭，无法建立 shell。
- 本地 heartbeat automation ID `240` 已设为 `PAUSED`，保留用户选择的每 30 分钟间隔。
- 结果、三种子汇总和审计：`reports/dependencies/finals_pre/minute_grid_240/additional_seeds_20260906/`。
- 原 seed 的评分和历史审计：`reports/dependencies/finals_pre/minute_grid_240/20260906/`。
- 新远程目录：`/root/autodl-tmp/minute-grid-240-moreseeds-20260906/`；总状态在 `results/status.json`，为 `completed`，`completed_seeds` 包含两个新增 seed。

240 位与 242 位平均中性化 RankIC 为 0.028562 / 0.028406，RankIC IR 为 0.684348 / 0.687640，多空 Sharpe 为 7.639835 / 7.388099，压力日 ICIR 为 0.394198 / 0.392953。Sharpe 三次配对都较高；其他指标两次较高、一次较低，ICIR 均值略低。完整标准差、配对差值和解释见[结果报告](../../../reports/dependencies/finals_pre/minute_grid_240/additional_seeds_20260906/README.md)。

三组源码/数据协议、配对内初始权重和训练顺序一致。六份逐日中性化 IC 的均值/ICIR、三个 seed 的汇总和 CSV 已在本地复核；四个新增 checkpoint 重新计算 SHA-256。原 seed 的大文件不在当前 checkout，使用先前回收审计及记录的 checkpoint 校验和，未声称本次重新校验其二进制。新增 checkpoint 和全量因子 parquet 留在本地结果目录，摘要、日志和审计证据纳入 Git。

242 基线是修改前 src 的固定时钟布局，不能称为原 submission 的尾部补空布局。三个 seed 覆盖相同 2024 年，不代表新增独立样本外时期。旧 submission 与正式历史成绩保留原定义。
