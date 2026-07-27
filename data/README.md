# 本地研究数据目录

本目录只保存经 AIStudio 按数据族聚合并通过官方界面下载的研究数据。原始分钟成交、盘口快照和未经筛选的财务全表不保存在本机。

聚合 Parquet 不进入 Git；Git 只保存本目录说明和 manifest。需要在规则允许的
同队成员之间同步时，按 `docs/team_workflow.md` 的“本地数据包同步”章节生成
带 SHA-256 的本地压缩包。原始平台导出、传输压缩包和临时文件必须放入
`data/raw/` 或 `data/transfers/`，这些内容不会进入 Git。

固定分层：

- `universe/`：历史股票池、交易状态和交易日历映射；
- `features/PV/`：日频量价通用组件；
- `features/MICRO_DAILY_FULL/`：分钟成交与盘口统一聚合后的2019—2023连续日频组件；
- `features/FR/`：严格 PIT 的财务状态或披露组件；
- `exposures/`：行业、市值、流动性和风险暴露；
- `labels/`：独立保存的未来收益标签；
- `factors/`：本地最小验证或平台回传的候选与组合结果；
  `candidate_pool.parquet` 是 `run_first_round.py` 生成的标准五列自研候选长表，
  可作为小型可复现输入同步给队友；其版本、行数和校验和记录在
  `manifest_candidate_pool.json`。

`features` 禁止包含未来标签或候选编号绑定的中间量。日频共享数据主键为 `date、instrument`；因子长表主键为 `date、instrument、candidate_id、factor_version`。Parquet 按数据族和年份分区，频率写入元数据而不是作为顶层目录。

各数据族的严格列集合、频率、可用时点、平台实测和财务事件键见 `docs/data_contract.md`；本机写入前使用 `bigalpha2026.feature_contracts.validate_feature_frame` 验收。

正式年度分区采用 `year=YYYY/part-YYYY.parquet`。每次传输必须附带 `manifest_YYYY.json`，并在本地核对 SHA-256、列、形状、主键、每日股票池数量和跨表键后才能删除传输压缩包。2019—2023 年基础包已经通过该流程；所有 manifest 均为 `evaluation_performed=false`，不得把数据验收解释为因子评价。

FR 事件面板按披露年份保存在 `features/FR/year=YYYY/part-YYYY.parquet`。历史与
2022 选择期清单为 `manifest_FR_2017_2022.json`，2023 确认期分区清单为
`manifest_FR_2023.json`。实际可用披露从 2018 年开始；2019 年首个交易日的
1,000 只股票均已有至少一条 LF 资产记录和两条 TTM 记录，因此可以在不使用未来
披露的前提下计算首次差分状态。

HF/OB 正式评价统一读取 `MICRO_DAILY_FULL`，不再读取代表月份或只供单个候选
使用的旧 OB 面板。平台原件保存为
`raw/MICRO_DAILY_FULL/micro_daily_YYYY.parquet`，随后运行
`scripts/prepare_micro_daily_full.py`，按历史股票池左连接并生成
`features/MICRO_DAILY_FULL/year=YYYY/part-YYYY.parquet`。标准面板保留
`micro_snapshot_available`；缺少分钟或盘口快照的组件保持 NaN，不在数据层
填0。五年源文件和标准文件的行数、日期覆盖、缺失键与SHA-256记录在
`manifest_MICRO_DAILY_FULL.json`。

旧 `features/HF/`、`features/OB/`、`features/OB_DAILY_FULL/` 及对应 manifest
只属于历史诊断，不得进入候选池的正式输入清单。

公开 36 因子库的完整离线参考快照保存在
`features/FACTORLIB_ALL36/year=YYYY/part-YYYY.parquet`，仅用于比赛 J
基础代理评分；screened15 训练输入仍保存在 `features/FACTORLIB/`。两者均使用
`run_combinations.py --check-files` 验收，默认的 `--check` 不读取比赛数据。
当前 screened15 快照在本地可观测为“每日截面均值约 0、标准差约 1”；manifest
将该事实记录为 `observed_value_scale`，但标准化究竟由平台表还是导出代码产生，
仍须在 AIStudio 确认，不能仅凭本地数值反推来源。
