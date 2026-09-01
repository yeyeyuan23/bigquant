# PRE experiment tests

这里的测试只验证实验设计和计算边界，不验证某个 IC 数字，也不运行训练。

在仓库根目录执行：

```bash
python -m pytest -q experiments/tests
```

覆盖范围：

- `test_global_contract.py`：O2C、241 日、42 个 Barra 回归量、三颗公共种子；
- `test_e1_walkforward.py`：60/20 日 schedule、一天标签隔离、未来前缀不变；
- `test_e2_epoch_curve.py`：六个 epoch 使用同一 OOS，RankIC 计算方向；
- `test_e3_progressive.py`：四臂结构、前向形状、无效配置；
- `test_e4_fixed_oos.py`：冻结边界、目标权重、换手与 bp 成本；
- `test_e5_raw_channels.py`：40 通道契约、17 通道基线切片、额外 23 通道隔离。
