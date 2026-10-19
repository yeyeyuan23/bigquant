"""Joint Candidate454 Elastic Net stress test for three M route ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

PROTOCOL = "candidate454_three_m_route_joint_diagnostic_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROUTE_NAMES = (
    "old_m",
    "m_dynamic_ensemble",
    "m_l5_ensemble",
)


def score_three_routes(
    score_reference: Any,
    routes: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    if tuple(routes) != EXPECTED_ROUTE_NAMES:
        raise ValueError(
            "three-route diagnostic requires ordered routes "
            f"{EXPECTED_ROUTE_NAMES}, got {tuple(routes)}"
        )
    return score_reference.score_joint_routes(routes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--route",
        nargs=2,
        action="append",
        metavar=("NAME", "PARQUET"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.score_submission_j_stability import (
        load_route,
        load_score_reference,
    )

    route_specs = [(str(name), Path(path)) for name, path in args.route]
    if tuple(name for name, _ in route_specs) != EXPECTED_ROUTE_NAMES:
        raise ValueError(
            "route order must be old_m, m_dynamic_ensemble, m_l5_ensemble"
        )
    routes: dict[str, pd.DataFrame] = {}
    route_sources: dict[str, str] = {}
    for name, path in route_specs:
        route, source = load_route(str(path), args.data_dir, (args.year,))
        routes[name] = route
        route_sources[name] = source

    score_reference = load_score_reference(
        args.data_dir,
        args.reports_dir,
        (args.year,),
    )
    scores = score_three_routes(score_reference, routes)
    rows = []
    for name in EXPECTED_ROUTE_NAMES:
        score = scores[name]
        rows.append(
            {
                "route": name,
                "J": float(score["score_proxy"]),
                "A": float(score["a_proxy"]),
                "B": float(score["b_proxy"]),
                "B_model_score": float(score["b_model_score"]),
                "B_mean_abs_weight": float(score["b_mean_abs_weight"]),
                "B_std_abs_weight": float(score["b_std_abs_weight"]),
                "B_nonzero_window_ratio": float(
                    score["b_nonzero_window_ratio"]
                ),
            }
        )

    output = {
        "protocol": PROTOCOL,
        "diagnostic_only": True,
        "note": (
            "This three-route joint Elastic Net is a sibling-crowding stress "
            "test. Primary ranking remains the individual Candidate454 plus "
            "one-route score."
        ),
        "year": args.year,
        "reference": score_reference.candidate454_metadata,
        "reference_factor_count": len(score_reference.reference_columns),
        "route_sources": route_sources,
        "routes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
