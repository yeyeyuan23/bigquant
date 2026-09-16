"""Reproduce displayed rules and split returns at calendar-year closing NAV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[3] / "data/runtime/finals_pre/e7_strategy_application/20260916/inputs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scenario", choices=["fees_slip0", "gross"], default="fees_slip0")
    args = parser.parse_args()
    inputs = args.inputs
    old_source = Path(__file__).resolve().parents[3] / "reports/dependencies/finals_pre/e7_strategy_application/20260916_simple_rules"
    comparison_path = Path(__file__).with_name("compare_simple_strategies.py")
    spec = importlib.util.spec_from_file_location("comparison", comparison_path)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    data = bt.load_inputs(inputs)
    prices = pd.read_parquet(inputs / "daily_prices.parquet")
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices["instrument"] = prices.instrument.astype(str)
    close = prices.pivot(index="date", columns="instrument", values="adjusted_close").reindex(
        index=data.dates, columns=data.instruments).to_numpy()
    close_missing = ~np.isfinite(close) | (close <= 0)
    close = np.where(close_missing, data.prices, close)
    cost = next(c for c in bt.SCENARIOS if c.scenario == args.scenario)
    old = pd.read_csv(old_source / "strategy_daily.csv")
    rows, summaries, checks = [], [], []
    for sid in ["baseline_daily", "buffer_20_30", "rank_mean_5d", "rebalance_2d", "rebalance_5d", "rank_mean_3d"]:
        if sid.startswith("rank_mean_"):
            window = int(sid.removeprefix("rank_mean_").removesuffix("d"))
            model_data = replace(data, scores=comparison.smooth_ranks(data.scores, window))
            engine_sid, every = "baseline_daily", 3
        elif sid.startswith("rebalance_"):
            model_data, engine_sid = data, "rebalance_3d"
            every = int(sid.removeprefix("rebalance_").removesuffix("d"))
        else:
            model_data, engine_sid, every = data, sid, 3
        daily, positions = bt.run(model_data, engine_sid, cost, capture_positions=True, rebalance_every=every)
        original = old[(old.strategy == sid) & (old.scenario == args.scenario) & (old.phase == 0)]
        np.testing.assert_allclose(daily.nav_end, original.nav_end, atol=1e-12, rtol=0)
        quantities = positions.pivot(index="entry_date", columns="instrument", values="adjusted_units")
        quantities = quantities.reindex(index=data.dates[1:-1].strftime("%Y-%m-%d"),
                                        columns=data.instruments).fillna(0).to_numpy()
        q = np.zeros(len(data.instruments))
        cash = 1.0
        nav = np.ones(len(data.dates))
        fees = np.zeros(len(data.dates))
        traded = np.zeros(len(data.dates))
        turnover = np.zeros(len(data.dates))
        missing_held = 0
        reconstruction_error = 0.0
        for j, t in enumerate(range(1, len(data.dates) - 1)):
            opening_nav = cash + np.dot(q, data.prices[t])
            np.testing.assert_allclose(opening_nav, daily.iloc[j].nav_start, atol=1e-11, rtol=0)
            trades = (quantities[j] - q) * data.prices[t]
            traded[t] = np.abs(trades).sum()
            turnover[t] = 0.5 * traded[t] / opening_nav
            fees[t] = cost.charge(trades)
            cash -= trades.sum() + fees[t]
            q = quantities[j]
            nav[t] = cash + np.dot(q, close[t])
            missing_held += int(((np.abs(q) > 1e-12) & close_missing[t]).sum())
            next_open_nav = cash + np.dot(q, data.prices[t + 1])
            if t == len(data.dates) - 2:
                terminal_trades = np.where(data.tradable[t + 1], -q * data.prices[t + 1], 0)
                traded[t + 1] = np.abs(terminal_trades).sum()
                turnover[t + 1] = 0.5 * traded[t + 1] / next_open_nav
                fees[t + 1] = cost.charge(terminal_trades)
                next_open_nav -= fees[t + 1]
                residual_q = np.where(data.tradable[t + 1], 0, q)
                # Untradable residual holdings remain marked using known quotes.
                np.testing.assert_allclose(np.abs(residual_q * data.prices[t + 1]).sum(),
                                           daily.iloc[j].terminal_residual_value, atol=1e-11, rtol=0)
                nav[t + 1] = next_open_nav + np.dot(residual_q, close[t + 1] - data.prices[t + 1])
                expected_cost = fees[t] + fees[t + 1]
            else:
                expected_cost = fees[t]
            np.testing.assert_allclose(expected_cost, daily.iloc[j].cost_amount, atol=1e-11, rtol=0)
            error = abs(next_open_nav - daily.iloc[j].nav_end)
            reconstruction_error = max(reconstruction_error, error)
            assert error < 1e-11
        returns = nav / np.r_[1.0, nav[:-1]] - 1
        np.testing.assert_allclose(np.prod(1 + returns), nav[-1], atol=1e-11, rtol=0)
        frame = pd.DataFrame({"strategy": sid, "date": data.dates.strftime("%Y-%m-%d"),
                              "closing_nav": nav, "net_return": returns, "cost_amount": fees,
                              "traded_notional": traded, "one_way_turnover": turnover})
        period_products = []
        for year in [2025, 2026]:
            mask = data.dates.year == year
            r = returns[mask]
            wealth = np.r_[1.0, np.cumprod(1 + r)]
            before_index = np.flatnonzero(mask)[0] - 1
            starting_nav = nav[before_index] if before_index >= 0 else 1.0
            ending_nav = nav[mask][-1]
            cumulative = ending_nav / starting_nav - 1
            np.testing.assert_allclose(cumulative, wealth[-1] - 1, atol=1e-12, rtol=0)
            period_products.append(1 + cumulative)
            summaries.append({"strategy": sid, "year": year, "days": int(mask.sum()),
                              "start": frame.loc[mask, "date"].iloc[0],
                              "end": frame.loc[mask, "date"].iloc[-1],
                              "start_nav": starting_nav, "end_nav": ending_nav,
                              "cumulative_net_return": cumulative,
                              "annualized_net_return": float(r.mean() * 252),
                              "average_one_way_turnover": float(turnover[mask].mean()),
                              "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
                              "cost_amount": float(fees[mask].sum())})
            frame.loc[mask, "period_nav"] = wealth[1:]
        np.testing.assert_allclose(np.prod(period_products), nav[-1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(fees.sum(), daily.cost_amount.sum(), atol=1e-11, rtol=0)
        np.testing.assert_allclose(traded.sum(), daily.traded_notional.sum(), atol=1e-10, rtol=0)
        rows.append(frame)
        checks.append({"strategy": sid, "holding_intervals_reproduced": len(daily),
                       "max_nav_reconstruction_error": reconstruction_error,
                       "missing_closing_quotes_on_held_names": missing_held,
                       "terminal_residual_value": float(daily.iloc[-1].terminal_residual_value),
                       "terminal_close_mark_change": float(nav[-1] - daily.iloc[-1].nav_end),
                       "yearly_compounding_reconciles": True, "total_cost_reconciles": True,
                       "total_traded_notional_reconciles": True})
        print(f"reproduced {sid}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    period_output = pd.DataFrame(summaries)
    daily_output = pd.concat(rows, ignore_index=True)
    if args.scenario == "gross":
        assert (period_output.cost_amount == 0).all()
        period_output = period_output.rename(columns={"cumulative_net_return": "cumulative_gross_return", "annualized_net_return": "annualized_gross_return"})
        daily_output = daily_output.rename(columns={"net_return": "gross_return"})
    period_output.to_csv(args.out / "period_returns.csv", index=False)
    daily_output.to_csv(args.out / "daily_closing_nav.csv", index=False)
    original_audit = json.loads((old_source / "audit.json").read_text())
    input_hashes = {p.name: sha(p) for p in inputs.glob("*.parquet")}
    assert input_hashes == original_audit["input_hashes"]
    assert sha(Path(bt.__file__)) == original_audit["engine_sha256"]
    report = {"status": "passed", "scenario": args.scenario, "buy_cost_bps": cost.buy_bps,
              "sell_cost_bps": cost.sell_bps, "slippage_bps": cost.slippage_bps, "checks": checks,
              "period_definition": "calendar years split at 2025-12-31 close; continuous holdings and trailing ranks",
              "closing_valuation": "adjusted close; missing closes use current opening mark or the engine's last known mark",
              "first_day": "2025-01-02: initial cash, zero return; first trade 2025-01-03 open",
              "last_day": "2026-08-28: original terminal liquidation at open; untradable residual holdings remain marked at close",
              "cumulative_return": "end NAV / preceding period end NAV - 1",
              "annualized_return": "mean of calendar-period daily closing-NAV returns * 252",
              "average_one_way_turnover": "mean over all trading days in the period of (buy plus sell notional) / (2 * pre-trade opening NAV); initial cash day zero; terminal trades assigned to their actual execution date",
              "selection_status": "retrospective; 2026 outcomes were already inspected when the five-day rule was chosen",
              "input_hashes": input_hashes, "engine_sha256": sha(Path(bt.__file__)),
              "comparison_script_sha256": sha(comparison_path), "script_sha256": sha(Path(__file__)),
              "output_hashes": {name: sha(args.out / name) for name in ["period_returns.csv", "daily_closing_nav.csv"]}}
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(period_output.to_string(index=False))


if __name__ == "__main__":
    main()
