# E6：17 通道的信息组消融

比较完整模型与四种通道删减版：去掉价格路径、去掉盘口、去掉成交结构、只保留价格路径和时钟。共五个配置、每个三个种子；完整模型使用 E3 的 P3 结果，其余四组单独训练。

## 代码入口

- [run.sh](run.sh)：启动四组删减版，传入要删除的通道组。
- [train_progressive.py](../e3_progressive_add/train_progressive.py)：执行通道删减和训练；E6 复用此程序，因此本目录没有第二份 `train.py`。
- [model_progressive.py](../e3_progressive_add/model_progressive.py)：模型网络结构。
- [score.py](score.py)：统一中性化评分，生成逐种子、汇总和配对差值表。

## 已完成的历史结果

| 配置 | RankIC 均值 | RankIC IR | 毛多空 Sharpe | 压力日 ICIR |
|---|---:|---:|---:|---:|
| 完整 17 通道 | 0.02787 | 0.672 | 7.35 | 0.385 |
| 去掉价格路径 | 0.02692 | 0.651 | 7.56 | 0.356 |
| 去掉盘口 | 0.02024 | 0.444 | 4.04 | 0.275 |
| 去掉成交结构 | 0.02471 | 0.596 | 6.39 | 0.414 |
| 只保留价格路径与时钟 | 0.02011 | 0.424 | 3.76 | 0.353 |

去掉盘口后四项指标在三个种子中都下降。价格路径和成交结构的贡献并非四项一致；这里没有计算置信区间或 p 值，多空 Sharpe 未扣交易成本。

原始 [逐种子结果](../../../reports/dependencies/finals_pre/e6_channel_groups/o2c/a4_per_seed.csv)、[均值及标准差](../../../reports/dependencies/finals_pre/e6_channel_groups/o2c/a4_summary.csv)、[配对差值](../../../reports/dependencies/finals_pre/e6_channel_groups/o2c/a4_paired_deltas.csv) 和 [评分审计](../../../reports/dependencies/finals_pre/e6_channel_groups/o2c/audit.json) 已归入主仓库。每个配置有三个种子、241 个评分日和 240,567 个有效股票日期样本。

这四份文件此前只收录在 PRE 材料仓库，主仓库缺少对应结果目录。此次按 GitHub 文件对象校验原样补入，重新计算了均值、样本标准差、16 项配对差值及同向次数；[补档核验记录](../../../reports/dependencies/finals_pre/e6_channel_groups/o2c/publication_audit.json)记录来源与校验和。本次没有重新训练或从原始因子重新评分。

## 历史结果与当前入口

上述是 2026-09-05 已发布到 PRE 的历史结果。对应历史代码参考 [c0dcd21](https://github.com/yeyeyuan23/bigquant/tree/c0dcd21/experiments/finals_pre/e6_channel_groups)，启动参数为 `--max-minutes 242`。

当前 `run.sh` 和 `score.py` 已改用 240 分钟，输出到 `o2c/clock240/`；本次补档不会把历史结果标成 240 分钟新实验。重跑当前入口前，需要准备相同协议下 E3 的完整模型基线。

比较 TCN 层数、分支数与核宽的 **15 配置 × 3 种子、45 次训练**属于 [E9 架构实验](../e9_tcn_architecture/README.md)。
