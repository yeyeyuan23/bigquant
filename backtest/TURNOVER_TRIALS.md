# Q10−Q1 换手优化：待执行方案

本轮由“降低换手，同时尽量保留收益”的要求发起。**状态：代码和固定方案已写好，AutoDL SSH 暂不可连接，尚未运行；没有新增回测结果。** 现有每日 Q10−Q1 扣费前年化 55.35%、Sharpe 3.206；扣费后年化 13.52%、Sharpe 0.784，日均单边换手 1.505。

保持模型权重、原始分数、行情、时点、2025-01-02 至 2026-08-28 区间、单边各 100% 目标敞口与费率不变。原始分数不做行业/风格中性化。不是重新训练，也不是保证收益。

## 一次固定的比较范围

共 21 个配置，每个重跑零成本、买 3bp/卖 8bp、每侧另加 2bp/5bp 四种情景，共 84 组。完整配置见 [protocol_turnover_trials.json](protocol_turnover_trials.json)。

| 方案 | 配置数 | 要回答的问题 |
| --- | ---: | --- |
| 每日 Q10−Q1 | 1 | 原结果能否逐日复现？ |
| 十分组排名缓冲 | 4 | 两端各持有 10%，保留带扩至 15% / 20% / 25% / 30%，能否减少边界附近的进出？ |
| 每 2 / 3 / 5 日调仓 | 10 | 减少交易次数后信号是否衰减？每个可能的交易日相位都跑，不能只挑有利起点。 |
| 每日渐进调仓 | 3 | 每次向最新目标调整 25% / 50% / 75%，能否平衡新信号和原持仓？ |
| 20% 缓冲 + 50% 渐进 | 1 | 两种降换手方式结合后的效果如何？ |
| 20% 缓冲 + 隔日调仓 | 2 | 两个相位是否都能保留收益？ |

缓冲方案保持每侧 100 只，但不再是每日严格的 Q10/Q1 成分；应称“基于十分组信号的排名缓冲”。渐进方案可持有超过 100 只，按多空方向分别归一至 +100% / −100%，不能把更低敞口带来的低换手当优化。降低频率时不交易日保持股数，敞口随价格漂移。所有相位首个可交易日均建仓，末日开盘均退出。

## 评价与边界

同时保留换手、毛/净年化、净 Sharpe、累计收益、最大回撤、成本敏感性，以及 2025 年和 2026 年分别的表现。公布全配置结果；不能在看到结果后静默增加搜索范围或只留下冠军。已看过整个评价区间，因此分年只用于描述稳定性，**不是独立的策略选择样本外检验**。本轮不把五日平均排名加回展示。

## AutoDL 运行

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/autodl-tmp/conda-envs/quant/bin/python -m unittest discover -s backtest -p 'test_*backtest.py' -v
/root/autodl-tmp/conda-envs/quant/bin/python -m unittest discover -s backtest -p 'test_turnover_trials.py' -v
/root/autodl-tmp/conda-envs/quant/bin/python backtest/run_turnover_trials.py \
  --raw-scores /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/m_raw_frozen_private_oos.parquet \
  --daily-prices /root/autodl-tmp/strategy-application-20260906/results/daily_prices.parquet \
  --inference-audit /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/e4_private_fixed_oos/inference_audit.json \
  --baseline-results backtest/results/20260917_decile_strategy_preview \
  --out backtest/results/20260917_turnover_trials
/root/autodl-tmp/conda-envs/quant/bin/python backtest/validate_turnover_trials.py backtest/results/20260917_turnover_trials
```

`status.json` 必须为 complete、84/84，并通过 audit.json。程序禁止在 macOS 运行回测；本地只写代码、复制已有小型展示数据、渲染 HTML。数值测试和独立账本复核均需远端执行。
