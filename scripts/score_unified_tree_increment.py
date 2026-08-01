"""Measure the paired J increment from adding the unified tree route."""

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
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2023])
    parser.add_argument("--tree-weights", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    years = tuple(args.years)
    baseline = normalized_route(pd.read_parquet(args.baseline))
    tree = normalized_route(pd.read_parquet(args.tree))
    baseline = baseline.loc[baseline["date"].dt.year.isin(years)]
    tree = tree.loc[tree["date"].dt.year.isin(years)]
    paired = baseline.rename(columns={"factor": "baseline"}).merge(
        tree.rename(columns={"factor": "tree"}),
        on=["date", "instrument"],
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError("neural and tree routes have no common OOS rows")
    paired["baseline"] = daily_rank(paired, "baseline")
    paired["tree"] = daily_rank(paired, "tree")
    baseline_route = paired[["date", "instrument", "baseline"]].rename(
        columns={"baseline": "factor"}
    )
    score_reference = load_score_reference(args.data_dir, args.reports_dir, years)

    rows: list[dict[str, object]] = []
    for tree_weight in args.tree_weights:
        if not 0.0 < tree_weight <= 1.0:
            raise ValueError("tree weights must be in (0, 1]")
        augmented = paired[["date", "instrument"]].copy()
        augmented["factor"] = daily_rank(
            paired.assign(
                blended=(1.0 - tree_weight) * paired["baseline"] + tree_weight * paired["tree"]
            ),
            "blended",
        )
        result = score_reference.paired_increment(
            baseline_route,
            augmented,
            include_stability=True,
        )
        rows.append({"tree_weight": tree_weight, **result})
        print(json.dumps(rows[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "paired_tree_increment_over_neural_v1",
        "baseline": str(args.baseline),
        "tree": str(args.tree),
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
