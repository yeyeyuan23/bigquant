# BigAlpha 2026 — 分钟微观结构因子

本仓库保存 BigAlpha 2026 决赛提交模型、数据合同、训练与推理代码，以及统一使用下一交易日开盘到收盘收益（O2C）的复盘实验。

正式提交模型只使用原始分钟行情构造的 17 个通道，不依赖 Candidate454 因子池。每天收盘后，模型读取当天分钟序列，为股票池中的每只股票输出一个无量纲横截面分数；分数用于预测下一交易日开盘到收盘的收益排序。

## 平台结果

正式提交 `6d0fe02c-1b61-41bd-9de9-fe9ac0774ffd`，checkpoint SHA256 为 `252c39ba…dcb`。

| 阶段 | 总分 | A | B |
|---|---:|---:|---:|
| 公榜 | **0.95480** | 0.95106 | 0.95641 |
| 私榜合并 | **0.88736** | 0.95788 | 0.85714 |

`总分 = 0.3 × A + 0.7 × B`。平台成绩与本地实验是两条证据链；本地结果不用于反推平台分数。

## 任务定义

对交易日 d：

1. 使用截至 d 日收盘的分钟数据；
2. 模型输出每只股票的无量纲 score；
3. 评价标签是下一交易日的 `close / open - 1`；
4. 每天计算中性化 score 排名与收益排名的 Spearman 相关；
5. 再跨交易日汇总 RankIC、RankIC IR、五分位多空 Sharpe 和压力日 RankIC IR。

中性化使用 10 个风格暴露和 32 个行业哑变量，共 42 个回归项。

## 模型

当前分钟输入、历史兼容与复现入口见 [240 分钟输入约定](docs/MINUTE_GRID.md)。

输入张量形状为 `[交易日批次, 股票, 分钟位置, 17 通道]`。模型使用 240 个固定分钟收盘时刻：09:31–11:30、13:01–15:00。缺失分钟留在原时刻，并由 `minute_mask` 标记；后续记录不会向前移动。17 个通道分为：

| 信息组 | 通道数 | 内容 |
|---|---:|---|
| 价格路径 | 3 | 分钟收益、振幅、收盘位置 |
| 盘口快照 | 5 | 相对价差、微价格偏移、不同档位深度不平衡与形状 |
| 成交结构 | 6 | 成交额、成交量、笔数、每笔强度和方向化成交额 |
| 时间位置 | 3 | 日内周期位置与上午/下午标记 |

网络包含两条并行通路：

- **序列通路**：标准化值与有效性标记拼接后，经 `Linear → GELU → LayerNorm` 投影到 96 维；三个 TCN 块依次处理序列。每个块并行使用 3、15、60 分钟因果卷积，立即混回 96 维并加残差。最后同时保留 `last`、`mean` 和 attention pooling 三种序列摘要。
- **统计通路**：每个通道计算均值、标准差、最后值、尾部 30 分钟均值和有效占比，得到 `17 × 5 = 85` 个日级统计量，再投影到 96 维。

两路各自形成 96 维后第一次融合，再通过 DeepSets 加入当天市场平均、市场分化和个股相对市场偏离，最后由打分头输出每只股票一个 score。

训练目标为：

```text
loss = -Pearson(prediction, target)
       + 0.05 × mean(SmoothL1(prediction, target))
```

训练 target 是下一交易日 O2C 收益在当天股票间的百分位排名，并缩放到 −1～1。Pearson 项让模型原始分数与该排名目标同向；SmoothL1 项约束分数尺度并降低少数大误差的影响。评估时使用 RankIC，而不是把训练用 Pearson 当作最终指标。

## 当前复盘证据

复盘实验统一使用 O2C 标签。完整协议、逐实验数字、限制和产物位置只维护在 [`experiments/finals_pre/README.md`](experiments/finals_pre/README.md)。当前已核验的 E1–E6 回答：

- expanding walk-forward 在 7 个未来 20 日窗口中都得到正 RankIC；
- 2019–2023 训练、2024 评价时，第 3 个 epoch 的四项均值最高；
- 渐进实验中，TCN 与 `last` 带来最大的 RankIC、RankIC IR 和 Sharpe 增量；
- 正式提交权重的预测力主要集中在下一交易日，五分位下一日收益保持单调；
- 每日五分位组合平均换手为 1.403，10bp 单边成本下收益和 Sharpe 转负；
- 通道组消融中，去掉盘口后四项均值和三个 seed 的配对结果一致下降；去掉成交结构后前三项一致下降，但压力 ICIR 不一致；
- 额外直接加入 23 个原始字段后，四项指标没有一致改善。

尚未完成评分和核验的实验不进入 README 结论。

## 主要目录

```text
src/alpha_models/
    microstructure.py                 当前 17 通道分钟模型
    temporal.py                       attention pooling 与 DeepSets

scripts/
    prepare_unified_microstructure_store.py
    train_unified_final_checkpoint.py
    evaluate_unified_microstructure.py
    build_unified_m_raw_submission.py

experiments/finals_pre/               当前 O2C 复盘实验与唯一结果 README
reports/dependencies/finals_pre/      大型实验产物与审计
submissions/                          冻结提交包
tests/                                特征、时间因果与模型测试
```

Candidate454、T/X 专家、Elastic Net 和旧 O2O/C2C 实验属于历史研究路径，不再作为当前提交模型或 PRE 证据。相关代码和结果可从 Git 历史与归档 tag 追溯。

## 数据与环境

代码仓库不保存分钟数据、完整因子 Parquet 或 checkpoint。AutoDL 运行时主要使用：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/data/` | 标签、股票池、暴露和小型清单 |
| `/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/` | 2019–2024 年 17 通道分钟训练 store |
| `/root/bigquant_private_data/` | 2025–2026 私榜分钟数据 |
| `/root/autodl-tmp/conda-envs/quant/bin/python` | 远端 Python / PyTorch 环境 |

## 验证

```bash
python -m ruff check .
coverage run -m pytest -q
coverage combine
coverage report --include="src/*"
```

时间因果检查位于 `tests/test_microstructure_expert.py` 和 `tests/test_microstructure_v2.py`，覆盖前缀不变性、午休分段、mask 和张量形状。平台成绩、冻结权重推理、本地评分和可交易回测是不同证据层；只有完成对应审计后才在 README 中写入结论。
