"""AutoDL-only fixed-grid exploration; all policies and calendar phases are reported."""

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from raw_engine import Costs, load_panel, run, summarize
from turnover_engine import run_rule


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if platform.system() != "Linux":
        raise SystemExit("Backtests and numerical audits must run on AutoDL, not the local Mac.")
    p = argparse.ArgumentParser()
    p.add_argument("--raw-scores", type=Path, required=True)
    p.add_argument("--daily-prices", type=Path, required=True)
    p.add_argument("--inference-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-results", type=Path, required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    protocol = json.loads((source / "protocol_turnover_trials.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    assert sha(args.raw_scores) == protocol["raw_scores_sha256"]
    assert sha(args.daily_prices) == protocol["daily_prices_sha256"]
    assert (
        json.loads(args.inference_audit.read_text())["checkpoint_sha256"]
        == protocol["checkpoint_sha256"]
    )
    data = load_panel(args.raw_scores, args.daily_prices, protocol["start"], protocol["end"])
    assert len(data.dates) == 402 and np.isfinite(data.scores).sum() == 402000
    started = datetime.now(UTC).isoformat()
    summaries, records, periods = [], [], []
    previous = pd.read_csv(args.baseline_results / "daily.csv")
    for rule in protocol["rules"]:
        for cost in protocol["costs"]:
            daily, _ = run_rule(data, rule, Costs(**cost))
            if rule["name"] == "daily_decile":
                legacy, _ = run(data, "daily_decile", Costs(**cost))
                pd.testing.assert_frame_equal(daily, legacy, check_exact=True)
                if cost["scenario"] in ("gross", "fees_slip0"):
                    old = previous[previous.scenario == cost["scenario"]]
                    np.testing.assert_allclose(
                        daily.closing_nav, old.closing_nav, atol=1e-12, rtol=0
                    )
            summaries.append(summarize(daily))
            records.append(daily)
            for year in ("2025", "2026"):
                d = daily[daily.date.str.startswith(year)].copy()
                # Returns retain their actual previous close at the calendar-year boundary.
                r = d.daily_return.to_numpy()
                curve = np.r_[1.0, np.cumprod(1 + r)]
                periods.append(
                    dict(
                        strategy=rule["name"],
                        scenario=cost["scenario"],
                        year=year,
                        days=len(d),
                        average_one_way_turnover=float(d.one_way_turnover.mean()),
                        annualized_return=float(r.mean() * 252),
                        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
                        cumulative_return=float(curve[-1] - 1),
                        max_drawdown=float((curve / np.maximum.accumulate(curve) - 1).min()),
                    )
                )
            (args.out / "status.json").write_text(
                json.dumps(dict(state="running", completed=len(summaries), total=84))
            )
        print(rule["name"], "complete", flush=True)
    summary = pd.DataFrame(summaries)
    daily = pd.concat(records, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    daily.to_csv(args.out / "daily.csv.gz", index=False, compression="gzip")
    pd.DataFrame(periods).to_csv(args.out / "calendar_years.csv", index=False)
    net = summary[summary.scenario == "fees_slip0"].copy()
    gross = summary[summary.scenario == "gross"].set_index("strategy")
    net["gross_annualized_return"] = net.strategy.map(gross.annualized_return)
    net["gross_sharpe"] = net.strategy.map(gross.sharpe)
    net.to_csv(args.out / "comparison.csv", index=False)
    daily[daily.scenario == "fees_slip0"][
        ["strategy", "date", "closing_nav", "daily_return", "one_way_turnover"]
    ].to_csv(args.out / "net_nav.csv", index=False)
    files = ["summary.csv", "daily.csv.gz", "calendar_years.csv", "comparison.csv", "net_nav.csv"]
    execution = dict(
        host=socket.gethostname(),
        platform=platform.platform(),
        python=sys.executable,
        started_utc=started,
        finished_utc=datetime.now(UTC).isoformat(),
        input_hashes=dict(raw_scores=sha(args.raw_scores), daily_prices=sha(args.daily_prices)),
        code_hashes={
            n: sha(source / n)
            for n in [
                "raw_engine.py",
                "turnover_engine.py",
                "run_turnover_trials.py",
                "test_turnover_trials.py",
                "validate_turnover_trials.py",
            ]
        },
        protocol_sha256=sha(source / "protocol_turnover_trials.json"),
        baseline_reproduced=True,
        output_hashes={n: sha(args.out / n) for n in files},
        neutralization=False,
        backtests_run_locally=False,
    )
    (args.out / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    (args.out / "status.json").write_text(
        json.dumps(dict(state="computed", completed=len(summaries), total=84)) + "\n"
    )
    print(
        net[
            ["strategy", "average_one_way_turnover", "annualized_return", "sharpe", "max_drawdown"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
