# E2：O2C epoch 曲线

用 2019–2023 expanding 历史训练，2024 保持 untouched OOS。三个 seed 各训练
6 个 epoch，并在每个 epoch 结束后保存对同一 241 日 OOS 的预测因子。所有
checkpoint 都用 `ret_next_open_to_close` 训练，并按 A 四小分比较。
