# E1：C2C walk-forward

从 2019 年开始使用 expanding 历史，每次训练后预测接下来的 20 个交易日；预测期覆盖 2023–2024。训练与评价统一使用 `ret_close_to_close`。

所有窗口的 2024 因子拼成一个严格 OOS 序列，再用完整 42 项 Barra 中性化计算 A 四小分。2024 完整 C2C 评价为 241 天。旧 O2C walk-forward 结果不参与本实验。
