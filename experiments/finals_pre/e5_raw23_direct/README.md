# E5：23 个原始字段直接通道（C2C）

使用已从 AIStudio 导出并在 AutoDL 核验的 `bigalpha_2026_stock_bar1m`
Parquet。2023 年训练、2024 年严格 OOS；三组相同 seed 比较原 17 个工程通道
与“17 + 23 个原始字段”共 40 通道。两臂都用 C2C 标签从零训练；网络、训练轮数和标签完全一致，唯一变化
是输入字段。

23 个直接通道为：open；二、三档买卖价；四、五档买卖价量；一至五档买卖
委托笔数。仅把无效报价记为缺失，不再构造组合特征。

标签使用仓库统一且已审计的 `ret_close_to_close`，严格按 484 个日文件的
`date/instrument` 键对齐。2024 年完整 C2C OOS 为 241 个可评分因子日。结果
写入 `reports/dependencies/finals_pre/e5_raw23_direct/c2c_autodl`，旧结果不参与比较。
