"""AutoDL-only drawdown and chart data for the executable Q10-Q1 strategy."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from raw_engine import targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if platform.system() != "Linux":
        raise SystemExit("Compute returns and drawdowns on AutoDL, not the local Mac.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--raw-scores", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    execution = json.loads((root / "execution.json").read_text())
    assert json.loads((root / "status.json").read_text())["validation"] == "passed"
    assert json.loads((root / "audit.json").read_text())["status"] == "passed"
    market_audit = json.loads((args.market / "audit.json").read_text())
    assert market_audit["status"] == "passed"
    for base, hashes in [(root, execution["output_hashes"]),
                         (args.market, market_audit["output_hashes"])]:
        for name, digest in hashes.items():
            assert sha(base / name) == digest
    assert sha(args.raw_scores) == execution["input_hashes"]["raw_scores"]
    raw = pd.read_parquet(args.raw_scores).sort_values(["date", "instrument"])
    checked = 0
    for _, block in raw.groupby("date"):
        scores = block.factor.to_numpy()
        assert len(scores) == 1000
        order = np.argsort(scores, kind="stable")
        weights = targets(scores, np.zeros(1000), groups=10)
        np.testing.assert_array_equal(np.sort(np.flatnonzero(weights < 0)), np.sort(order[:100]))
        np.testing.assert_array_equal(np.sort(np.flatnonzero(weights > 0)), np.sort(order[-100:]))
        np.testing.assert_allclose(weights[weights > 0], .01, atol=0, rtol=0)
        np.testing.assert_allclose(weights[weights < 0], -.01, atol=0, rtol=0)
        checked += 1
    assert checked == 402

    daily = pd.read_csv(root / "daily.csv")
    net = daily[(daily.strategy == "daily_decile") & (daily.scenario == "fees_slip0")].copy()
    gross = daily[(daily.strategy == "daily_decile") & (daily.scenario == "gross")].copy()
    market = pd.read_csv(args.market / "market_reference.csv")
    assert net.date.tolist() == gross.date.tolist() == market.date.tolist()
    assert len(net) == 402 and net.closing_nav.iloc[0] == market.closing_nav.iloc[0] == 1
    frames, metrics = [], {}
    summary = pd.read_csv(root / "summary.csv")
    for sid, frame in [("daily_decile", net), ("csi1000_reference", market)]:
        nav = frame.closing_nav.to_numpy()
        assert np.isfinite(nav).all() and (nav > 0).all()
        returns = nav / np.r_[1., nav[:-1]] - 1
        np.testing.assert_allclose(returns, frame.daily_return, atol=1e-12, rtol=0)
        high = np.maximum.accumulate(np.r_[1., nav])[1:]
        drawdown = nav / high - 1
        # Independently audit every drawdown by walking the historical high water mark.
        running_high = 1.
        for i, value in enumerate(nav):
            running_high = max(running_high, value)
            assert abs(drawdown[i] - (value - running_high) / running_high) < 1e-14
        trough = int(np.argmin(drawdown))
        peak = int(np.argmax(nav[:trough + 1]))
        recovered = np.flatnonzero(nav[trough + 1:] >= high[trough])
        recovery = int(trough + 1 + recovered[0]) if len(recovered) else None
        stats = {
            "start": frame.date.iloc[0], "end": frame.date.iloc[-1], "days": len(nav),
            "annualized_return": float(returns.mean() * 252),
            "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)),
            "cumulative_return": float(nav[-1] - 1),
            "cagr": float(nav[-1] ** (252 / len(nav)) - 1),
            "max_drawdown": float(drawdown[trough]),
            "max_drawdown_peak": frame.date.iloc[peak],
            "max_drawdown_trough": frame.date.iloc[trough],
            "max_drawdown_recovery": frame.date.iloc[recovery] if recovery is not None else None,
        }
        if sid == "daily_decile":
            stats["average_one_way_turnover"] = float(net.one_way_turnover.mean())
            stats["gross_annualized_return"] = float(summary[summary.scenario == "gross"].annualized_return.iloc[0])
            expected = summary[summary.scenario == "fees_slip0"].iloc[0]
        else:
            expected = json.loads((args.market / "summary.json").read_text())
        for key in ("annualized_return", "sharpe", "cumulative_return", "cagr", "max_drawdown"):
            np.testing.assert_allclose(stats[key], expected[key], atol=1e-12, rtol=0)
        metrics[sid] = stats
        frames.append(pd.DataFrame({
            "date": frame.date, "series": sid, "closing_nav": nav,
            "daily_return": returns, "cumulative_return_pct": (nav - 1) * 100,
            "running_peak_nav": high, "drawdown_pct": drawdown * 100,
            "one_way_turnover": frame.one_way_turnover if sid == "daily_decile" else np.nan,
        }))
    pd.concat(frames, ignore_index=True).to_csv(root / "preview_daily.csv", index=False)
    (root / "preview_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    receipt = {
        "status": "passed", "host": socket.gethostname(),
        "finished_utc": datetime.now(UTC).isoformat(), "score_group_checks": checked,
        "chart_rows": 804, "drawdown_independent_checks": 804,
        "strategy": "Daily Q10-Q1, +100% and -100%, net of buy 3bp/sell 8bp; overnight included",
        "market": "CSI 1000 price index, market reference only; never subtracted",
        "cumulative": "100 * (compounded closing NAV - 1); not arithmetic O2C accumulation",
        "drawdown": "100 * (closing NAV / historical maximum NAV including initial 1 - 1)",
        "calendar_note": market_audit["calendar_note"],
        "input_hashes": {"daily.csv": sha(root / "daily.csv"),
                         "market_reference.csv": sha(args.market / "market_reference.csv")},
        "code_sha256": sha(__file__),
        "output_hashes": {n: sha(root / n) for n in ["preview_daily.csv", "preview_summary.json"]},
    }
    (root / "preview_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("PASS: 402 independent group checks and 804 drawdown checks.")


if __name__ == "__main__":
    main()
