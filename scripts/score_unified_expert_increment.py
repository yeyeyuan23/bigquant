"""Measure paired J increment over the Candidate462 Elastic Net baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from score_submission_j_stability import load_score_reference, normalized_route


def daily_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date")[column].rank(pct=True) * 2.0 - 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--expert-name", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--expert-weights", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    years = tuple(args.years)
    baseline = normalized_route(pd.read_parquet(args.baseline))
    expert = normalized_route(pd.read_parquet(args.expert))
    baseline = baseline.loc[baseline["date"].dt.year.isin(years)]
    expert = expert.loc[expert["date"].dt.year.isin(years)]
    paired = baseline.rename(columns={"factor": "baseline"}).merge(
        expert.rename(columns={"factor": "expert"}),
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("baseline and expert routes have no common OOS rows")
    paired["baseline"] = daily_rank(paired, "baseline")
    paired["expert"] = daily_rank(paired, "expert")
    baseline_route = paired[["date", "instrument", "baseline"]].rename(
        columns={"baseline": "factor"}
    )
    score_reference = load_score_reference(args.data_dir, args.reports_dir, years)

    rows: list[dict[str, object]] = []
    for expert_weight in args.expert_weights:
        if not 0.0 < expert_weight <= 1.0:
            raise ValueError("expert weights must be in (0, 1]")
        augmented = paired[["date", "instrument"]].copy()
        augmented["factor"] = daily_rank(
            paired.assign(
                blended=(1.0 - expert_weight) * paired["baseline"]
                + expert_weight * paired["expert"]
            ),
            "blended",
        )
        result = score_reference.paired_increment(
            baseline_route,
            augmented,
            include_stability=True,
        )
        rows.append({"expert_weight": expert_weight, **result})
        print(json.dumps(rows[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "paired_expert_increment_over_candidate462_elasticnet_v1",
        "baseline_role": "candidate462_full_pool_elasticnet_oos",
        "baseline": str(args.baseline),
        "expert": str(args.expert),
        "expert_name": args.expert_name,
        "years": list(years),
        "common_rows": len(paired),
        "common_days": int(paired["date"].nunique()),
        "results": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
