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
- `features/FR/`：严格 PIT 的财务状态或披露组件；
- `exposures/`：行业、市值、流动性和风险暴露；
- `labels/`：独立保存的未来收益标签；
- `factors/`：本地生成的候选和组合长表；`candidate_pool.parquet` 是
  `run_first_round.py` 自动生成的统一自研因子入口。

`features` 禁止包含未来标签或候选编号绑定的中间量。日频共享数据主键为 `date、instrument`；因子长表主键为 `date、instrument、candidate_id、factor_version`。Parquet 按数据族和年份分区，频率写入元数据而不是作为顶层目录。

各数据族的严格列集合、频率、可用时点、平台实测和财务事件键见 `docs/data_contract.md`；本机写入前使用 `bigalpha2026.feature_contracts.validate_feature_frame` 验收。

正式年度分区采用 `year=YYYY/part-YYYY.parquet`。每次传输必须附带 `manifest_YYYY.json`，并在本地核对 SHA-256、列、形状、主键、每日股票池数量和跨表键后才能删除传输压缩包。2019—2023 年基础包已经通过该流程；所有 manifest 均为 `evaluation_performed=false`，不得把数据验收解释为因子评价。

FR 事件面板按披露年份保存在 `features/FR/year=YYYY/part-YYYY.parquet`。开发历史清单为 `manifest_FR_2017_2022.json`，2023 样本外分区清单为 `manifest_FR_2023.json`。实际可用披露从 2018 年开始；2019 年首个交易日的 1,000 只股票均已有至少一条 LF 资产记录和两条 TTM 记录，因此可以在不使用未来披露的前提下计算首次差分状态。

HF/OB 日频共享面板按 `features/HF|OB/year=YYYY/month=MM/part-YYYY-MM.parquet` 保存，每月对应 `manifest_HFOB_YYYY-MM.json`。默认首轮读取 2019—2023 每年 2、8 月，共 10 个必跑月份。因子计算前的市场状态检查发现开发样本缺少低流动性尾部，故按冻结规则唯一启用 `2022-11`；该选择未读取任何候选值。其余 7 个已下载备选月不参与本轮，2023 年 5、11 月尚未生成。

公开 36 因子库固定放在
`features/FACTORLIB/year=YYYY/part-YYYY.parquet`。组合脚本按合同动态读取全部特征
列，并以 `universe/` 为左表；因子库或自研因子缺失不会再通过内连接删掉股票日。
