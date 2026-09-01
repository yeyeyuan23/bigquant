# E5：23 个原始字段直接通道（O2C）

baseline17 使用正式 17 通道；raw40 在完全相同设置下只增加 23 个原始字段，
合计 40 通道。两臂都使用 `ret_next_open_to_close` 从零训练；网络、训练轮数、
seed 和评分方式一致，唯一变化是输入通道。

标签按 484 个日文件的 `date/instrument` 键与权威 2023–2024 标签对齐。2024
完整 O2C OOS 为 241 个可评分因子日。结果写入
`reports/dependencies/finals_pre/e5_raw23_direct/o2c_autodl`。
