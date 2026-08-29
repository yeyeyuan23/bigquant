# E2：C2C epoch 曲线

2019–2023 expanding 训练，2024 untouched OOS。三个 seed 各训练 6 个 epoch，并在每个 epoch 结束后保存同一 241 日预测因子。所有 checkpoint 都用 `ret_close_to_close` 训练，并按 A 四小分比较。

逐 epoch 因子由通用训练器的 `--eval-every-epoch` 路径保存，和最终因子使用同一预测函数，避免第二套实现漂移。
