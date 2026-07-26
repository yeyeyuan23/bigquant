# 本地研究数据目录

本目录只保存经 AIStudio 按数据族聚合并通过官方界面下载的研究数据。原始分钟成交、盘口快照和未经筛选的财务全表不保存在本机。

聚合 Parquet 不进入 Git；Git 只保存本目录说明和 manifest。需要在规则允许的
同队成员之间同步时，按 `docs/team_workflow.md` 的“本地数据包同步”章节生成
带 SHA-256 的本地压缩包。原始平台导出、传输压缩包和临时文件必须放入
`data/raw/` 或 `data/transfers/`，这些内容不会进入 Git。

固定分层：

- `universe/`：历史股票池、交易状态和交易日历映射；
- `features/PV/`：日频量价通用组件；
- `features/HF/`：分钟成交聚合后的日频通用组件；
- `features/OB/`：分钟盘口聚合后的日频通用组件；
- `features/OB_DAILY_FULL/`：供 `OB-001` 使用的 2019—2023 完整年度盘口日频组件；
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

HF/OB 日频共享面板按 `features/HF|OB/year=YYYY/month=MM/part-YYYY-MM.parquet` 保存，每月对应 `manifest_HFOB_YYYY-MM.json`。默认首轮读取 2019—2023 每年 2、8 月，共 10 个必跑月份。因子计算前的市场状态检查发现开发样本缺少低流动性尾部，故按冻结规则唯一启用 `2022-11`；该选择未读取任何候选值。其余 7 个已下载备选月不参与本轮，2023 年 5、11 月尚未生成。

所有 HF/OB 月度 manifest 必须记录对应 HF、OB 文件的路径、SHA-256、shape、
严格列集合和主键检查。文件同步或修复后运行
`PYTHONPATH=src python scripts/refresh_data_manifests.py` 统一复核并刷新这些字段。

完整 OB 日频底座不再使用上述代表月份。平台导出的全市场聚合原件保存在
`raw/OB_DAILY_FULL/ob_daily_YYYY.parquet`，随后运行
`scripts/prepare_ob_daily_full.py`，按历史股票池裁成
`features/OB_DAILY_FULL/year=YYYY/part-YYYY.parquet`。标准面板保留明确的
`ob_snapshot_available` 标志；缺少盘口快照的组件保持 NaN，不在数据层填 0。
原始分钟盘口仍留在 AIStudio。源文件与标准文件的行数、日期覆盖、缺失键和
SHA-256 均记录在 `manifest_OB_DAILY_FULL.json`。

公开 36 因子库和大型训练矩阵默认留在 AIStudio；本目录只接收评价报告、特征
重要性和预测结果，不要求为本地训练下载完整副本。只有需要离线复现时，才按
`features/FACTORLIB/year=YYYY/part-YYYY.parquet` 保存可选快照，并使用
`run_combinations.py --check-files` 验收；默认的 `--check` 不读取比赛数据。
当前 screened15 快照在本地可观测为“每日截面均值约 0、标准差约 1”；manifest
将该事实记录为 `observed_value_scale`，但标准化究竟由平台表还是导出代码产生，
仍须在 AIStudio 确认，不能仅凭本地数值反推来源。
