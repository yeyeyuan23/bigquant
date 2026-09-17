# 回测源码与固定协议

[回测总览](../README.md) · [运行命令](../docs/TURNOVER_TRIALS.md) · [实验结果索引](../results/README.md)。

原 `backtest/` 根目录的 Python 源码、测试和协议整体放在这里，保留文件内容、相对导入和 `__file__` 查找协议的关系。运行命令从仓库根目录出发，使用 `python backtest/src/<脚本>.py`。数值回测与测试在 AutoDL 执行。

## 代码用途

| 文件 | 用途 |
| --- | --- |
| [raw_engine.py](raw_engine.py) | 原始分数输入、目标持仓、含费用成交、缺报价处理、逐日账本与指标 |
| [turnover_engine.py](turnover_engine.py) | 排名缓冲、低频调仓、渐进混合及组合规则 |
| [run_turnover_trials.py](run_turnover_trials.py) | 固定运行全部 21 × 4，校验输入并复现每日基线 |
| [run_raw_scores.py](run_raw_scores.py) | 历史三规则、纯多头和每日十分组的运行入口，由 `--protocol` 选择 |
| [validate_turnover_trials.py](validate_turnover_trials.py) | 从保存账本独立核对 84 组与 168 组分年汇总 |
| [validate_results.py](validate_results.py) | 原始分数系列的独立账本与指标核对 |
| [audit_turnover_candidate.py](audit_turnover_candidate.py) | 指定候选的持仓成员和未来输入扰动检查 |
| [report_turnover_trials.py](report_turnover_trials.py) | 读取已审计结果，生成比较、相位范围和原始结果 README |
| [evaluate_raw_deciles.py](evaluate_raw_deciles.py) / [evaluate_raw_quintiles.py](evaluate_raw_quintiles.py) | 因子十分组 / 五分位的下一日开盘到收盘评价 |
| [build_decile_strategy_preview.py](build_decile_strategy_preview.py) | 每日十分组净值、收益和回撤展示数据 |
| [build_market_reference.py](build_market_reference.py) | 中证 1000 双源行情与股票池核验 |
| [test_raw_backtest.py](test_raw_backtest.py) | 14 项费用、会计、报价、敞口和信号时序测试 |
| [test_turnover_trials.py](test_turnover_trials.py) | 8 项基线复现、排名缓冲、调仓相位、渐进与因果测试 |
| [diagnostics/compare_score_inputs.py](diagnostics/compare_score_inputs.py) | 原始分数与历史处理流程的同口径诊断；独立的历史补充实验 |

## 每批协议

| 协议 | 持仓配置 | 成本数 | 运行数 | 结果目录 |
| --- | ---: | ---: | ---: | --- |
| [protocol_turnover_trials.json](protocol_turnover_trials.json) | 21 | 4 | 84 | [20260917_turnover_trials](../results/20260917_turnover_trials/README.md) |
| [protocol.json](protocol.json) | 3 | 4 | 12 | [20260916_raw_scores](../results/20260916_raw_scores/README.md) |
| [protocol_long_only.json](protocol_long_only.json) | 1 | 4 | 4 | [20260917_long_only_buffer](../results/20260917_long_only_buffer/README.md) |
| [protocol_decile.json](protocol_decile.json) | 1 | 2 | 2 | [20260917_decile_strategy_preview](../results/20260917_decile_strategy_preview/README.md) |

每个结果目录内的 `protocol.json` 是当次运行副本；不要把它与这里默认的三规则 `protocol.json` 混淆。分组评价与市场参照使用各自命令行参数及审计记录，不计入这些持仓回测运行数。
