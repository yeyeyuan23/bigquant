# 决赛复盘实验（finals_pre）

这里保存 PRE 当前仍使用的实验代码、协议和产物位置；结果数字统一写在
`BigAlpha_Pre/experiment_log.md`，不在两处重复维护。

## 统一收益口径

- 新训练和新评价只使用 `ret_close_to_close`（C2C）。
- 日期为 d 的因子，对应下一交易日的 `close / pre_close - 1`；严格按市场交易日映射，个股停牌时记为缺失，不顺延期限。
- 2024 年有 242 个交易日，最后一天没有下一交易日标签，因此完整 C2C OOS 为 241 个可评分因子日。
- 实验表只报告 A 四小分：Barra 中性化 RankIC、RankIC IR、五分位多空 Sharpe、压力日 RankIC IR。B、N、J 不再用于实验选择。
- 旧 O2O/O2C 实验可以作为开发历史保留，但不得与 C2C 主表混算，也不得继续沿用原编号。

唯一例外是 E4：它研究正式提交模型在私榜期的衰减，因此必须保留“checkpoint 历史上用 O2C 训练”这一事实；该模型不再训练，只按统一 C2C 口径评价。

## 当前编号

| 编号 | 问题 | 代码目录 | 当前 C2C 产物 |
|---|---|---|---|
| **E1** | walk-forward：滚动重训是否优于长期冻结 | `e1_c2c_walkforward/` | `e1_c2c_walkforward/`；旧结果不进主表 |
| **E2** | epoch 曲线：训练轮数如何影响严格 OOS | `e2_c2c_epoch_curve/` | `e2_c2c_epoch_curve/`；旧结果不进主表 |
| **E3** | 渐进加法：统计基线 → DeepSets → TCN+last → 完整三摘要 | `e15_progressive_add/` | `e15_progressive_add/c2c/` |
| **E4** | 冻结正式提交权重在 2025–2026 私榜期的表现与换手成本 | `e16_private_fixed_oos/` | `e16_private_fixed_oos/` |
| **E5** | 23 个原始字段直接作为通道是否优于原 17 通道 | `e17_raw23_direct/` | AIStudio：`/home/aiuser/work/e17_raw23_results_c2c/`，完成后回传 |

目录名保留旧编号是为了避免破坏脚本引用；PRE 对外材料只使用上表的新编号。

## 当前运行边界

- E3 的四个模型臂必须全部用同一 C2C 标签从零训练。完整模型不能复用旧 O2O 因子。
- E5 在 AIStudio 本地读取 `bigalpha_2026_stock_bar1m`、构建存储并训练；历史分钟原始数据不复制到 Mac 或 AutoDL。
- E4 使用私榜 `bigalpha_2026_stock_bar15m_private`，冻结 checkpoint sha256 为 `252c39ba...dcb`；推理代码不生成其他收益标签。
- E1、E2 在 C2C 重跑完成前没有可进入 PRE 主表的正式结果。

## 目录约定

```text
experiments/finals_pre/<实验目录>/           可复现代码和 README
reports/dependencies/finals_pre/<实验目录>/  AutoDL 结果和审计
reports/dependencies/finals_pre/<实验目录>/c2c/  与旧口径并存时的 C2C 独立产物
```

旧 E1–E14 的置换、删减、N 分、454 因子池、O2O 标签和结构筛选结果仍可在原目录追溯，但已经退出当前 PRE 证据链。除非明确做 C2C 重训，不应把它们重新写回主实验表。
