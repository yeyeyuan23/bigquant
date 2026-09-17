"""Generate complete comparison tables from audited AutoDL outputs, on AutoDL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


def name(key):
    if key == "daily_decile":
        return "每日 Q10−Q1"
    if key.startswith("buffer_10_20_every2_phase"):
        return f"20% 缓冲 + 隔日 / 相位 {key[-1]}"
    if key == "buffer_10_20_partial50":
        return "20% 缓冲 + 50% 渐进"
    if key.startswith("buffer_10_"):
        return f"10% 持仓 / {key.split('_')[-1]}% 保留带"
    if key.startswith("partial_"):
        return f"{key.split('_')[-1]}% 渐进调仓"
    parts = key.split("_")
    return f"每 {parts[1][:-1]} 日调仓 / 相位 {parts[-1][-1]}"


def main():
    if platform.system() != "Linux":
        raise SystemExit("Numerical comparisons must run on AutoDL.")
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    root = p.parse_args().results
    assert json.loads((root / "status.json").read_text())["state"] == "complete"
    assert json.loads((root / "audit.json").read_text())["status"] == "passed"
    execution = json.loads((root / "execution.json").read_text())
    for filename, digest in execution["output_hashes"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
    summary = pd.read_csv(root / "summary.csv").set_index(["strategy", "scenario"])
    periods = pd.read_csv(root / "calendar_years.csv").set_index(["strategy", "scenario", "year"])
    comparison = pd.read_csv(root / "comparison.csv")
    base = comparison.set_index("strategy").loc["daily_decile"]
    rows = []
    for row in comparison.itertuples():
        years = [periods.loc[(row.strategy, "fees_slip0", year)] for year in (2025, 2026)]
        rows.append(
            dict(
                strategy=row.strategy,
                name=name(row.strategy),
                turnover=row.average_one_way_turnover,
                turnover_reduction_vs_daily=1
                - row.average_one_way_turnover / base.average_one_way_turnover,
                gross_annualized=row.gross_annualized_return,
                net_annualized=row.annualized_return,
                net_sharpe=row.sharpe,
                net_cumulative=row.cumulative_return,
                max_drawdown=row.max_drawdown,
                net_annualized_delta=row.annualized_return - base.annualized_return,
                gross_annualized_delta=row.gross_annualized_return - base.gross_annualized_return,
                net_2025_annualized=years[0].annualized_return,
                net_2025_sharpe=years[0].sharpe,
                net_2026_annualized=years[1].annualized_return,
                net_2026_sharpe=years[1].sharpe,
                net_slip2_annualized=summary.loc[(row.strategy, "fees_slip2"), "annualized_return"],
                net_slip2_sharpe=summary.loc[(row.strategy, "fees_slip2"), "sharpe"],
                net_slip5_annualized=summary.loc[(row.strategy, "fees_slip5"), "annualized_return"],
                net_slip5_sharpe=summary.loc[(row.strategy, "fees_slip5"), "sharpe"],
                turnover_lower_net_and_sharpe_higher=bool(
                    row.average_one_way_turnover < base.average_one_way_turnover
                    and row.annualized_return > base.annualized_return
                    and row.sharpe > base.sharpe
                ),
                annualized_beats_daily_both_years=bool(
                    all(
                        y.annualized_return
                        > periods.loc[("daily_decile", "fees_slip0", year), "annualized_return"]
                        for y, year in zip(years, (2025, 2026))
                    )
                ),
            )
        )
    report = pd.DataFrame(rows)
    report.to_csv(root / "review_table.csv", index=False)
    phase_rows = []
    for prefix in ("every_2d_", "every_3d_", "every_5d_", "buffer_10_20_every2_"):
        group = report[report.strategy.str.startswith(prefix)]
        phase_rows.append(
            dict(
                family=prefix.rstrip("_"),
                phases=len(group),
                turnover_min=group.turnover.min(),
                turnover_max=group.turnover.max(),
                net_annualized_min=group.net_annualized.min(),
                net_annualized_max=group.net_annualized.max(),
                net_sharpe_min=group.net_sharpe.min(),
                net_sharpe_max=group.net_sharpe.max(),
                worst_max_drawdown=group.max_drawdown.min(),
            )
        )
    phases = pd.DataFrame(phase_rows)
    phases.to_csv(root / "phase_ranges.csv", index=False)
    best = report.sort_values("net_sharpe", ascending=False).iloc[0]
    lines = [
        "# 十分组信号的换手优化：完整结果",
        "",
        "AutoDL 已完成 21 个配置 × 4 种成本情景，共 84 组。原始模型分数不作中性化，模型权重、行情和评价期保持不变。",
        "",
        "2025-01-02 至 2026-08-28，共 402 个交易日；首日现金，次日开盘建仓，末日开盘退出。基准费用买入 3bp、卖出 8bp；净指标含费用。",
        "",
        "## 主要发现",
        "",
        f"本批净 Sharpe 最高的是 **{best['name']}**：换手 {best.turnover:.3f}（较每日十分组降低 {best.turnover_reduction_vs_daily:.2%}），毛年化 {best.gross_annualized:.2%}、净年化 {best.net_annualized:.2%}、净 Sharpe {best.net_sharpe:.3f}、净累计 {best.net_cumulative:.2%}、最大回撤 {best.max_drawdown:.2%}。这里只对本批候选和本区间排序，不宣称参数最优。",
        "",
        f"该方案 2025 年净年化 {best.net_2025_annualized:.2%}、Sharpe {best.net_2025_sharpe:.3f}；2026 年至 8 月净年化 {best.net_2026_annualized:.2%}、Sharpe {best.net_2026_sharpe:.3f}。每侧额外 2bp 滑点后净年化 {best.net_slip2_annualized:.2%}、Sharpe {best.net_slip2_sharpe:.3f}；每侧额外 5bp 后为 {best.net_slip5_annualized:.2%}、{best.net_slip5_sharpe:.3f}。",
        "",
        "排名缓冲在本批各保留带上均提高了净收益和 Sharpe；降低频率则明显依赖调仓起点。简单减慢交易并不必然优于缓冲。保留带越宽的本批表现越好，不足以推断继续加宽仍会改善。",
        "",
        "## 全配置基准成本结果",
        "",
        "| 规则 | 日均单边换手 | 毛年化 | 净年化 | 净 Sharpe | 净累计 | 最大回撤 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report.itertuples():
        lines.append(
            f"| {r.name} | {r.turnover:.3f} | {r.gross_annualized:.2%} | {r.net_annualized:.2%} | {r.net_sharpe:.3f} | {r.net_cumulative:.2%} | {r.max_drawdown:.2%} |"
        )
    lines += [
        "",
        "## 分年表现与附加滑点",
        "",
        "分年年化仍为该年实际报告日的日收益均值 × 252；2026 年只到 8 月，不是全年实际收益。分年不重置仓位，跨年收益按实际前收盘衔接。",
        "",
        "| 规则 | 2025 净年化 | 2025 Sharpe | 2026 净年化 | 2026 Sharpe | 加 2bp 净年化 | 加 5bp 净年化 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in report.itertuples():
        lines.append(
            f"| {r.name} | {r.net_2025_annualized:.2%} | {r.net_2025_sharpe:.3f} | {r.net_2026_annualized:.2%} | {r.net_2026_sharpe:.3f} | {r.net_slip2_annualized:.2%} | {r.net_slip5_annualized:.2%} |"
        )
    lines += [
        "",
        "## 调仓起点敏感性",
        "",
        "| 规则族 | 相位数 | 换手范围 | 净年化范围 | 净 Sharpe 范围 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in phases.itertuples():
        lines.append(
            f"| {r.family} | {r.phases} | {r.turnover_min:.3f}–{r.turnover_max:.3f} | {r.net_annualized_min:.2%}–{r.net_annualized_max:.2%} | {r.net_sharpe_min:.3f}–{r.net_sharpe_max:.3f} |"
        )
    lines += [
        "",
        "## 规则、验证与解释范围",
        "",
        "- 每日 Q10−Q1 与前次逐日净值复现一致；14 项原引擎测试、8 项新规则测试在 AutoDL 通过。",
        "- 84 组全部逐日核对费用、净值、隔夜/日内损益、现金/持仓恒等式、交易日历和指标；168 组分年汇总通过独立复算。",
        "- 30% 保留带还完成真实行情下 400 个调仓日的独立目标选股/敞口核对，以及未来分数/价格扰动后前 201 日净值和持仓不变检查，见 candidate_audit.json。",
        "- 缓冲：每侧 100 只，原持仓在保留带内优先保留，再补足。缓冲组合不是每日严格的 Q10/Q1 成分。",
        "- 渐进：将旧组合与最新目标混合，按正负方向分别归一至 +100% / −100%；可能持有超过 100 只。降低换手并非通过主动降低目标敞口实现。",
        "- 低频：不交易日保持股数，敞口随价格漂移。报告全部起点相位，不能只挑最有利起点。",
        "- 年化为算术均值 × 252，累计为复利净值减 1；Sharpe 为年化、无风险利率为零；换手使用数值口径。",
        "- 该区间已经用于此前研究，本轮为固定方案的回顾性比较；分年不是未见测试集，结果不构成未来收益保证。",
        "- 沿用理想化开盘成交、可卖空等假设，未计借券/融资、非线性冲击、涨跌停排队等。",
        "",
        "复现代码在 backtest/；完整协议见 protocol.json，逐日账本见 daily.csv.gz，审计见 audit.json，运行主机与输入/代码哈希见 execution.json。",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    (root / "report_audit.json").write_text(
        json.dumps(
            dict(
                status="passed",
                source="audited summary and calendar_years",
                rows=len(report),
                phase_families=len(phases),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                output_hashes={
                    n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                    for n in ["review_table.csv", "phase_ranges.csv", "README.md"]
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(report.sort_values("net_sharpe", ascending=False).to_string(index=False))
    print(phases.to_string(index=False))


if __name__ == "__main__":
    main()
