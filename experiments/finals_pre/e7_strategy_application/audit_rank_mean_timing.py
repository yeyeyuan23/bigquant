"""Audit the saved five-session rank strategy without changing its results."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[3] / "data/runtime/finals_pre/e7_strategy_application/20260916/inputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3] / "reports/dependencies/finals_pre/e7_strategy_application/20260916_simple_rules"
    inputs = args.inputs
    script = Path(__file__).with_name("compare_simple_strategies.py")
    spec = importlib.util.spec_from_file_location("comparison", script)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    original = json.loads((source / "audit.json").read_text())
    hashes = {p.name: digest(p) for p in inputs.glob("*.parquet")}
    assert hashes == original["input_hashes"]
    assert digest(Path(bt.__file__)) == original["engine_sha256"]
    assert digest(script) == original["script_sha256"]

    data = bt.load_inputs(inputs)
    signal_frame = pd.read_parquet(inputs / "signals.parquet")
    signal_dates = pd.DatetimeIndex(sorted(pd.to_datetime(signal_frame.date).unique()))
    assert data.dates.equals(signal_dates)
    assert data.dates.is_monotonic_increasing and data.dates.is_unique
    assert (data.dates.dayofweek < 5).all()
    smoothed = comparison.smooth_ranks(data.scores, 5)
    ranks = pd.DataFrame(data.scores).rank(axis=1, method="average", pct=True).to_numpy()
    counts = np.zeros_like(data.scores, dtype=int)
    manual_max_error = 0.0
    for signal_index in range(len(data.dates)):
        block = ranks[max(0, signal_index - 4):signal_index + 1]
        count = np.isfinite(block).sum(axis=0)
        expected = np.divide(np.nansum(block, axis=0), count,
                             out=np.full(len(data.instruments), np.nan), where=count > 0)
        expected[~np.isfinite(data.scores[signal_index])] = np.nan
        counts[signal_index] = count
        np.testing.assert_allclose(smoothed[signal_index], expected, rtol=0, atol=1e-14,
                                   equal_nan=True)
        manual_max_error = max(manual_max_error,
                               float(np.nanmax(np.abs(smoothed[signal_index] - expected))))

    cost = next(c for c in bt.SCENARIOS if c.scenario == "fees_slip0")
    full, positions = bt.run(replace(data, scores=smoothed), "baseline_daily", cost,
                             capture_positions=True)
    saved = pd.read_csv(source / "strategy_daily.csv")
    saved = saved[(saved.strategy == "rank_mean_5d") & (saved.scenario == "fees_slip0")
                  & (saved.phase == 0)].reset_index(drop=True)
    assert len(full) == len(saved) == 400
    numeric = ["nav_start", "nav_end", "net_return", "cost_amount", "one_way_turnover"]
    np.testing.assert_allclose(full[numeric], saved[numeric], rtol=0, atol=1e-12)
    for key in ["signal_date", "entry_date", "exit_date"]:
        assert full[key].equals(saved[key])
    assert full.signal_date.tolist() == data.dates[:-2].strftime("%Y-%m-%d").tolist()
    assert full.entry_date.tolist() == data.dates[1:-1].strftime("%Y-%m-%d").tolist()

    checks = []
    for cutoff in [1, 4, 5, 20, 100, 200, len(data.dates) - 3]:
        # At the cutoff opening, only raw signals strictly before cutoff exist.
        prefix_scores = comparison.smooth_ranks(data.scores[:cutoff], 5)
        np.testing.assert_allclose(prefix_scores, smoothed[:cutoff], rtol=0, atol=0,
                                   equal_nan=True)
        end = cutoff + 2  # one extra opening for the engine's terminal marking
        prefix = replace(data, dates=data.dates[:end], scores=smoothed[:end],
                         prices=data.prices[:end], tradable=data.tradable[:end])
        prefix_daily, prefix_positions = bt.run(prefix, "baseline_daily", cost,
                                               capture_positions=True)
        day = str(data.dates[cutoff].date())
        expected_positions = positions[positions.entry_date <= day].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix_positions, expected_positions, check_exact=True)
        # Exclude the terminal-return row, whose forced liquidation is intentional.
        pd.testing.assert_frame_equal(prefix_daily.iloc[:-1], full.iloc[:cutoff - 1],
                                      check_exact=True)

        future_scores = data.scores.copy()
        future_scores[cutoff:] *= -100
        future_scores[cutoff:, ::13] = np.nan
        altered_smooth = comparison.smooth_ranks(future_scores, 5)
        np.testing.assert_allclose(altered_smooth[:cutoff], smoothed[:cutoff], rtol=0,
                                   atol=0, equal_nan=True)
        future_prices = data.prices[:end].copy()
        future_prices[cutoff + 1:] *= 1.01
        altered = replace(prefix, scores=altered_smooth[:end], prices=future_prices)
        altered_daily, altered_positions = bt.run(altered, "baseline_daily", cost,
                                                  capture_positions=True)
        pd.testing.assert_frame_equal(altered_positions, expected_positions, check_exact=True)
        pd.testing.assert_frame_equal(altered_daily.iloc[:-1], full.iloc[:cutoff - 1],
                                      check_exact=True)
        checks.append({"entry_index": cutoff, "entry_date": day,
                       "signal_prefix_equal": True, "position_prefix_equal": True,
                       "future_score_and_price_invariance": True})
        print(f"passed cutoff {day}", flush=True)

    windows = []
    for entry_index in range(1, len(data.dates) - 1):
        signal_index = entry_index - 1
        count = counts[signal_index]
        eligible = np.isfinite(smoothed[signal_index])
        target = bt.targets(smoothed[signal_index], np.zeros(len(data.instruments)),
                            "baseline_daily")
        selected = np.abs(target) > 0
        windows.append({"entry_date": str(data.dates[entry_index].date()),
                        "window_start": str(data.dates[max(0, entry_index - 5)].date()),
                        "window_end": str(data.dates[signal_index].date()),
                        "window_trading_days": min(5, entry_index),
                        "eligible_names": int(eligible.sum()),
                        "eligible_with_fewer_than_five_ranks": int((eligible & (count < 5)).sum()),
                        "selected_names": int(selected.sum()),
                        "selected_with_fewer_than_five_ranks": int((selected & (count < 5)).sum())})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(windows).to_csv(args.out / "rank_windows.csv", index=False)
    summary = bt.summarize(full)
    report = {
        "status": "passed", "scope": "five-session smoothing and next-session execution on frozen inputs",
        "trade_day_t_uses": "cross-sectional percentile ranks from trading dates t-5 through t-1",
        "warmup": "available dates used during first four entries",
        "missing_ranks": "mean of finite ranks in the same five-session window; prior-session score required",
        "dates": len(data.dates), "holding_intervals": len(full),
        "manual_window_max_error": manual_max_error,
        "saved_daily_max_error": float(np.max(np.abs(full[numeric].to_numpy() - saved[numeric].to_numpy()))),
        "cutoff_checks": checks,
        "summary": {k: summary[k] for k in ["annualized_return", "average_one_way_turnover", "max_drawdown"]},
        "warmup_entry_days": sum(w["window_trading_days"] < 5 for w in windows),
        "selected_stock_day_records_with_short_windows_after_warmup": sum(
            w["selected_with_fewer_than_five_ranks"] for w in windows if w["window_trading_days"] == 5),
        "boundaries": ["Upstream factor inference and historical data publication/revision timestamps were not rerun.",
                       "Selecting the five-day rule after viewing this evaluation period remains retrospective strategy selection."],
        "input_hashes": hashes, "engine_sha256": digest(Path(bt.__file__)),
        "comparison_script_sha256": digest(script), "audit_script_sha256": digest(Path(__file__)),
    }
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
