"""Execute on AutoDL: python backtest/run_raw_scores.py --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from raw_engine import NAMES, Costs, load_panel, run, smooth_ranks, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if platform.system() != "Linux":
        raise SystemExit("This task's backtests must run on AutoDL (Linux), not the local Mac.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--daily-prices", type=Path, required=True)
    parser.add_argument("--inference-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    protocol = json.loads((source / "protocol.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n")
    (args.out / "status.json").write_text(json.dumps({"state": "running", "completed": 0}))
    assert sha(args.raw_scores) == protocol["raw_scores_sha256"], "Wrong raw factor file"
    assert sha(args.daily_prices) == protocol["daily_prices_sha256"], "Wrong daily price file"
    inference = json.loads(args.inference_audit.read_text())
    assert inference["checkpoint_sha256"] == protocol["checkpoint_sha256"]
    started = datetime.now(UTC).isoformat()
    data = load_panel(args.raw_scores, args.daily_prices, protocol["start"], protocol["end"])
    raw = pd.read_parquet(args.raw_scores)
    expected = raw.pivot(index="date", columns="instrument", values="factor").reindex(
        index=data.dates, columns=data.instruments
    ).to_numpy()
    np.testing.assert_allclose(data.scores, expected, rtol=0, atol=0, equal_nan=True)
    assert len(data.dates) == 402
    assert np.isfinite(data.scores).sum() == 402000
    # A future score change must not affect earlier smoothed signals/positions.
    cutoff = 200
    smoothed = smooth_ranks(data.scores)
    perturbed = data.scores.copy()
    perturbed[cutoff:] *= -100
    np.testing.assert_allclose(smooth_ranks(perturbed)[:cutoff], smoothed[:cutoff],
                               rtol=0, atol=0, equal_nan=True)
    summaries, records = [], []
    for strategy in protocol["strategies"]:
        for config in protocol["costs"]:
            daily, _ = run(data, strategy, Costs(**config))
            summaries.append(summarize(daily))
            records.append(daily)
            (args.out / "status.json").write_text(json.dumps(
                {"state": "running", "completed": len(summaries), "total": 12}))
            print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(summaries)
    daily = pd.concat(records, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    daily.to_csv(args.out / "daily.csv", index=False)
    primary = []
    for strategy in protocol["strategies"]:
        gross = summary[(summary.strategy == strategy) & (summary.scenario == "gross")].iloc[0]
        net = summary[(summary.strategy == strategy) & (summary.scenario == "fees_slip0")].iloc[0]
        primary.append({
            "strategy": strategy, "name": NAMES[strategy], "start": net.start, "end": net.end,
            "days": int(net.days), "average_one_way_turnover": net.average_one_way_turnover,
            "gross_annualized_return": gross.annualized_return,
            "net_annualized_return": net.annualized_return,
            "net_sharpe": net.sharpe, "gross_sharpe": gross.sharpe,
            "net_cagr": net.cagr, "net_cumulative_return": net.cumulative_return,
            "net_max_drawdown": net.max_drawdown,
        })
    pd.DataFrame(primary).to_csv(args.out / "primary.csv", index=False)
    daily[daily.scenario == "fees_slip0"][["strategy", "date", "closing_nav", "daily_return",
                                            "one_way_turnover"]].to_csv(
        args.out / "net_nav.csv", index=False)
    code_hashes = {p.name: sha(p) for p in sorted(source.glob("*.py"))
                   if not p.name.startswith("._")}
    execution = {
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.executable,
        "python_version": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "pyarrow": version("pyarrow"),
        "started_utc": started, "finished_utc": datetime.now(UTC).isoformat(),
        "raw_scores_path": str(args.raw_scores), "daily_prices_path": str(args.daily_prices),
        "inference_audit_sha256": sha(args.inference_audit),
        "input_hashes": {"raw_scores": sha(args.raw_scores), "daily_prices": sha(args.daily_prices)},
        "code_hashes": code_hashes, "protocol_sha256": sha(source / "protocol.json"),
        "raw_score_values_preserved_exactly": True, "neutralization": False,
        "winsorization": False, "standardization": False,
        "raw_factor_rows": int(np.isfinite(data.scores).sum()), "sessions": len(data.dates),
        "holding_intervals": len(data.dates) - 2, "first_trade": str(data.dates[1].date()),
        "last_liquidation": str(data.dates[-1].date()), "future_score_invariance": True,
        "output_hashes": {p.name: sha(p) for p in sorted(args.out.glob("*.csv"))},
    }
    (args.out / "execution.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2)+"\n")
    (args.out / "status.json").write_text(json.dumps({"state": "computed", "completed": 12}))


if __name__ == "__main__":
    main()
