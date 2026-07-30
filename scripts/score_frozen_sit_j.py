"""Score frozen S/I/T route memberships without rerunning admission.

This entrypoint deliberately treats the three frozen report files as inputs.
It trains only the final S, I and T route models, then scores their outputs
against one shared competition-score reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bigalpha2026.combinations import (
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
)
from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    family_balanced_factor,
    j_baseline_columns_from_self_columns,
    load_dynamic_inputs,
    prepare_experiment_context,
)


def _load_members(path: Path, key: str) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} contains no {key}")
    members = tuple(
        value if str(value).startswith("self__") else f"self__{value}"
        for value in values
    )
    if len(members) != len(set(members)):
        raise ValueError(f"{path} contains duplicate {key}")
    return members


def _membership_digest(routes: dict[str, tuple[str, ...]]) -> str:
    payload = {route: list(members) for route, members in routes.items()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/runtime/all156_full_20260729"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/runtime_all156_full_20260729"),
    )
    parser.add_argument("--years", nargs="+", type=int, default=[2023])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/runtime_all156_full_20260729/"
            "j_frozen_sit_independent"
        ),
    )
    args = parser.parse_args()

    latest = args.reports_dir / "latest"
    routes = {
        "S": _load_members(latest / "s_only_result.json", "admitted_candidates"),
        "I": _load_members(latest / "i_only_result.json", "frozen_candidates"),
        "T": _load_members(
            latest / "t_orthogonal_only_result.json",
            "tree_admitted_candidates",
        ),
    }
    requested_candidates = tuple(
        sorted({member for members in routes.values() for member in members})
    )
    years = tuple(dict.fromkeys((*DEVELOPMENT_YEARS, *args.years)))
    (
        panel,
        labels,
        exposures,
        _coverage,
        _candidate_pool,
        all36_reference,
        single_factor_candidates,
    ) = load_dynamic_inputs(
        args.data_dir,
        args.reports_dir,
        candidate_filter=requested_candidates,
        years=years,
        include_exposures=False,
        include_all36=True,
    )
    public_columns = tuple(
        column for column in panel.columns if column.startswith("factorlib__")
    )
    self_columns = tuple(
        column for column in panel.columns if column.startswith("self__")
    )
    missing = {
        route: sorted(set(members).difference(self_columns))
        for route, members in routes.items()
        if set(members).difference(self_columns)
    }
    if missing:
        raise ValueError(f"frozen route members are absent from panel: {missing}")
    j_baseline_columns = j_baseline_columns_from_self_columns(self_columns)
    (
        oriented,
        selected_public,
        _screening,
        _j_reference_directions,
        score_reference,
        _s_candidate_features,
    ) = prepare_experiment_context(
        panel,
        labels,
        exposures,
        all36_reference,
        public_columns,
        self_columns,
        j_baseline_columns,
        single_factor_candidates,
    )

    prediction_years = tuple(args.years)
    s_factor = family_balanced_factor(oriented, routes["S"])
    i_factor, i_weights = walk_forward_elastic_net_with_weights(
        oriented,
        labels,
        feature_columns=(*selected_public, *routes["I"]),
        prediction_years=prediction_years,
    )
    t_factor = walk_forward_lightgbm(
        oriented,
        labels,
        feature_columns=(*selected_public, *routes["T"]),
        prediction_years=prediction_years,
        residual_baseline_columns=selected_public,
    )
    raw = {"S": s_factor, "I": i_factor, "T": t_factor}
    scored_routes: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for route, factor in raw.items():
        block = factor.loc[
            pd.to_datetime(factor["date"]).dt.year.isin(prediction_years)
        ].copy()
        if block.empty:
            raise ValueError(f"{route} produced no rows for {prediction_years}")
        direction_score = score_reference.score_best_direction(block)
        direction = float(direction_score["selected_direction"])
        block["factor"] = pd.to_numeric(
            block["factor"], errors="coerce"
        ) * direction
        scored_routes[route] = block
        score = score_reference.score(block)
        rows.append(
            {
                "route": route,
                "member_count": len(routes[route]),
                "years": ",".join(str(year) for year in prediction_years),
                "selected_direction": direction,
                "J_base": float(score["score_proxy"]),
                "A_base": float(score["a_proxy"]),
                "B_base": float(score["b_proxy"]),
            }
        )

    common_dates = set.intersection(
        *(
            set(pd.to_datetime(frame["date"]).dt.normalize().unique())
            for frame in scored_routes.values()
        )
    )
    if not common_dates:
        raise ValueError("S/I/T routes have no common dates")
    crowding = score_reference.score_joint_routes(
        {
            route: frame.loc[frame["date"].isin(common_dates)]
            for route, frame in scored_routes.items()
        }
    )
    for row in rows:
        crowded = crowding[str(row["route"])]
        row["J_crowded"] = float(crowded["score_proxy"])
        row["A_crowded"] = float(crowded["a_proxy"])
        row["B_crowded"] = float(crowded["b_proxy"])
        row["J_robust"] = min(
            float(row["J_base"]),
            float(row["J_crowded"]),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(
        ["J_robust", "J_base"],
        ascending=False,
    )
    summary.to_csv(args.output_dir / "frozen_sit_j_summary.csv", index=False)
    i_weights.to_csv(args.output_dir / "i_elastic_net_weights.csv", index=False)
    output = {
        "status": "ok",
        "protocol": "frozen_sit_independent_J_v1",
        "note": (
            "Scores frozen report memberships as-is; admission and frozen "
            "state compatibility checks are intentionally not rerun."
        ),
        "data_dir": str(args.data_dir),
        "reports_dir": str(args.reports_dir),
        "years": list(prediction_years),
        "development_years": list(DEVELOPMENT_YEARS),
        "membership_digest": _membership_digest(routes),
        "members": {route: list(members) for route, members in routes.items()},
        "summary": summary.to_dict("records"),
    }
    (args.output_dir / "frozen_sit_j_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
