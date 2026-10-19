"""Score route parquets on fixed half-year periods with the local J proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_submission_j_stability import load_score_reference, normalized_route


def period_bounds(year: int, half: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    half = half.lower()
    if half == "h1":
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 6, 30)
    if half == "h2":
        return pd.Timestamp(year, 7, 1), pd.Timestamp(year, 12, 31)
    raise ValueError(f"unsupported half: {half}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", nargs="+", type=Path)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--lambda-std", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_score_reference(
        args.data_dir,
        args.reports_dir,
        tuple(args.years),
    )
    results = []
    for route_path in args.routes:
        route = normalized_route(pd.read_parquet(route_path))
        periods = []
        for year in args.years:
            for half in ("h1", "h2"):
                start, end = period_bounds(year, half)
                block = route.loc[route["date"].between(start, end)].copy()
                if block.empty:
                    raise ValueError(f"{route_path} contains no rows for {year}_{half}")
                score = reference.score(block)
                periods.append(
                    {
                        "period": f"{year}_{half}",
                        "rows": len(block),
                        "days": int(block["date"].nunique()),
                        "A": float(score["a_proxy"]),
                        "B": float(score["b_proxy"]),
                        "J": float(score["score_proxy"]),
                    }
                )
        values = np.asarray([row["J"] for row in periods], dtype=float)
        results.append(
            {
                "route": str(route_path),
                "periods": periods,
                "J_mean": float(values.mean()),
                "J_worst": float(values.min()),
                "J_std": float(values.std()),
                "J_stable": float(values.mean() - args.lambda_std * values.std()),
            }
        )
    payload = {
        "protocol": "half_year_local_J_proxy_v1",
        "evidence_boundary": "Local proxy only; not a platform score.",
        "results": results,
        "lambda_std": args.lambda_std,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
