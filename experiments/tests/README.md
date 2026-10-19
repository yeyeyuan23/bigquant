# PRE experiment tests

这里的测试验证实验设计、实际数据变换和计算边界，不固定某个正式 IC 数字，
也不启动正式训练。小型数据会执行生产标签构造、完整 Barra 中性化和 A 分评分链路。

在仓库根目录执行：

```bash
python -m pytest -q experiments/tests
```

覆盖范围：

- `test_global_contract.py`：O2C、完整 Barra schema、三颗公共种子和评分项 wiring；
- `test_label_alignment.py`：下一交易日 O2C 的真实构造路径；同日、跨两日和未来泄漏负测；
- `test_scoring_pipeline.py`：10 风格 + 32 行业逐列中性化、完整 E5 A 分链路及缺列/换序/多列/重复键负测；
- `test_e1_walkforward.py`：60/20 日 schedule、一天标签隔离、未来前缀不变；
- `test_e2_epoch_curve.py`：六个 epoch 使用同一 OOS，RankIC 计算方向；
- `test_e3_progressive.py`：四臂结构、前向形状、无效配置、按 seed 相同的日期与股票样本；
- `test_e4_fixed_oos.py`：冻结边界、目标权重、换手与 bp 成本；
- `test_e5_raw_channels.py`：40 通道逐项名称/顺序、17 个公式数值、换序负测、按 seed 相同 shuffle、额外 23 通道隔离。
