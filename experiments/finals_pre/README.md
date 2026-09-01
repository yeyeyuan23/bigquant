# 决赛复盘实验（finals_pre）

这里保存 PRE 当前仍使用的实验代码、协议和产物位置；结果数字统一写在
`BigAlpha_Pre/experiment_log.md`，不在两处重复维护。

## 统一收益口径

- 新训练和新评价只使用 `ret_next_open_to_close`（O2C）。
- 日期为 d 的因子对应下一交易日的 `close / open - 1`；严格按市场交易日映射，个股停牌时记为缺失，不顺延期限。
- 2024 年有 242 个交易日，最后一天没有下一交易日标签，因此完整 O2C OOS 为 241 个可评分因子日。
- 实验表只报告 A 四小分：Barra 中性化 RankIC、RankIC IR、五分位多空 Sharpe、压力日 RankIC IR。B、N、J 不用于实验选择。
- 旧 C2C/O2O 结果不进入当前主表；完成 O2C 重跑并核验后删除旧正式产物。

E4 使用历史正式提交 checkpoint。该 checkpoint 本来就是 O2C 训练权重；实验不更新权重，只在 2025–2026 数据上做严格 O2C 样本外评价。

## 当前编号

| 编号 | 问题 | 代码目录 | 当前 O2C 产物 |
|---|---|---|---|
| **E1** | walk-forward：滚动重训是否优于长期冻结 | `e1_o2c_walkforward/` | `e1_o2c_walkforward/` |
| **E2** | epoch 曲线：训练轮数如何影响严格 OOS | `e2_o2c_epoch_curve/` | `e2_o2c_epoch_curve/` |
| **E3** | 渐进加法：统计基线 → DeepSets → TCN+last → 完整三摘要 | `e3_progressive_add/` | `e3_progressive_add/o2c/` |
| **E4** | 冻结正式提交权重在 2025–2026 私榜期的表现与换手成本 | `e4_private_fixed_oos/` | `e4_private_fixed_oos/` |
| **E5** | 23 个原始字段直接作为通道是否优于原 17 通道 | `e5_raw23_direct/` | `e5_raw23_direct/o2c_autodl/` |

## 运行边界

- E1 从 2019 年历史开始扩展训练，私榜期每增加 60 个交易日从零重训，预测紧接着的 20 日。
- E2 用 2019–2023 训练，保存三个 seed 每个 epoch 对同一 2024 OOS 的预测。
- E3 四个模型臂全部使用同一 O2C 标签从零训练，不能复用旧口径因子。
- E5 两臂只允许 23 个原始字段通道这一处差异，并使用同一 O2C 标签从零训练。
- E4 冻结 checkpoint sha256 为 `252c39ba...dcb`；推理不重训，评分使用私榜 1 分钟数据生成的严格下一交易日 O2C 标签。

## 目录约定

```text
experiments/finals_pre/<实验目录>/           可复现代码和 README
reports/dependencies/finals_pre/<实验目录>/  AutoDL 结果和审计
reports/dependencies/finals_pre/e3_progressive_add/o2c/  渐进加法 O2C 产物
```

旧置换、删减、N 分、454 因子池和结构筛选代码不占用当前实验编号；历史可从 Git 提交记录追溯。
