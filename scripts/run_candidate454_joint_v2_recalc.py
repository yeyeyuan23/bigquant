"""Recalculate a Candidate454 joint official-period proxy with full diagnostics.

Each group is scored once over its complete continuous evaluation period.  Year
is only used to select source rows, not as an official scoring or averaging
unit.  The report also records the exact exposure panel used by the local
style/industry neutralization proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_submission_j_stability import (
    file_sha256,
    load_score_reference,
    normalized_route,
)

KEY_COLUMNS = ("date", "instrument")
PROTOCOL = "candidate454_joint_submission_period_proxy_v4"
REPORT_SCHEMA_VERSION = 4
OFFICIAL_BARRA_STYLE_COLUMNS = (
    "SIZE",
    "BETA",
    "MOMENTUM",
    "RESVOL",
    "SIZENL",
    "BTOP",
    "LIQUIDTY",
    "EARNYILD",
    "GROWTH",
    "LEVERAGE",
)
A_COMPONENTS = (
    "rank_ic_mean",
    "rank_ic_ir",
    "long_short_sharpe",
    "stress_ic_ir",
)
SCORE_DETAIL_COLUMNS = (
    "score_proxy",
    "a_proxy",
    "b_proxy",
    *(f"a_{component}" for component in A_COMPONENTS),
    *(f"a_{component}_percentile" for component in A_COMPONENTS),
    "b_model_score",
    "b_mean_abs_weight",
    "b_std_abs_weight",
    "b_nonzero_window_ratio",
    "score",
)
RUN_AUDIT_COLUMNS = (
    "score_days",
    "score_weight_windows",
    "reference_factor_count",
    "joint_route_count",
    "joint_common_rows",
)
DETAIL_COLUMNS = (*SCORE_DETAIL_COLUMNS, *RUN_AUDIT_COLUMNS)
FULL_RESULT_COLUMNS = (
    "group",
    "route",
    "family",
    "years",
    "J",
    "A",
    "B",
    *DETAIL_COLUMNS,
)
SUMMARY_RESULT_COLUMNS = (
    "group",
    "rank",
    "route",
    "family",
    "years",
    "J",
    "A",
    "B",
    *DETAIL_COLUMNS,
)


def daily_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Map each day's cross-section to the scorer's minus-one-to-one ranks."""

    numeric = pd.to_numeric(values, errors="coerce")
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    grouped = numeric.groupby(normalized_dates, sort=False)
    counts = grouped.transform("count")
    ranked = grouped.rank(method="average")
    result = ((ranked / counts) - 0.5) * 2.0
    return result.astype(float)


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _route_audit(route: pd.DataFrame) -> dict[str, Any]:
    values = route["factor"].to_numpy(dtype="float64")
    daily_unique = route.groupby("date", sort=False)["factor"].nunique()
    return {
        "rows": len(route),
        "days": int(route["date"].nunique()),
        "date_min": str(route["date"].min().date()),
        "date_max": str(route["date"].max().date()),
        "years": sorted(map(int, route["date"].dt.year.unique())),
        "duplicate_keys": int(route.duplicated(list(KEY_COLUMNS)).sum()),
        "nonfinite_factor_rows": int((~np.isfinite(values)).sum()),
        "constant_days": int(daily_unique.le(1).sum()),
    }


def _load_parts(spec: dict[str, Any], root: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    parts = tuple(_resolve(path, root) for path in spec["parts"])
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"route {spec['name']} parts missing: {missing}")
    frames = [pd.read_parquet(path) for path in parts]
    route = normalized_route(pd.concat(frames, ignore_index=True))
    sources = []
    for path in parts:
        try:
            audited_path = str(path.relative_to(root))
        except ValueError:
            audited_path = str(path)
        sources.append({"path": audited_path, "sha256": file_sha256(path)})
    return route, sources


def _weighted_rank_route(
    spec: dict[str, Any],
    routes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    components = spec["components"]
    if not components:
        raise ValueError(f"formula route {spec['name']} has no components")
    merged: pd.DataFrame | None = None
    provenance = []
    for index, component in enumerate(components):
        source_name = str(component["route"])
        if source_name not in routes:
            raise KeyError(
                f"formula route {spec['name']} references unknown route {source_name}"
            )
        weight = float(component["weight"])
        column = f"factor_{index}"
        frame = routes[source_name].copy()
        frame["factor"] = daily_rank(frame["factor"], frame["date"])
        frame = frame.rename(columns={"factor": column})
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
        )
        provenance.append(
            {
                "route": source_name,
                "weight": weight,
                "column": column,
                "input_preprocessing": "daily_cross_sectional_rank_minus_one_to_one",
            }
        )
    assert merged is not None
    raw = sum(
        item["weight"] * merged[item["column"]]
        for item in provenance
    )
    merged["factor"] = daily_rank(raw, merged["date"])
    return normalized_route(merged), provenance


def materialize_routes(
    manifest: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    output_dir = (
        output_dir.resolve()
        if output_dir.is_absolute()
        else (root / output_dir).resolve()
    )
    routes: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    specs = manifest["routes"]
    names = [str(spec["name"]) for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("route names in manifest are not unique")

    for spec in specs:
        if spec.get("kind", "parquet") != "parquet":
            continue
        name = str(spec["name"])
        route, sources = _load_parts(spec, root)
        routes[name] = route
        metadata[name] = {
            "name": name,
            "family": str(spec["family"]),
            "kind": "parquet",
            "sources": sources,
            "audit": _route_audit(route),
        }

    for spec in specs:
        if spec.get("kind", "parquet") != "weighted_daily_rank":
            continue
        name = str(spec["name"])
        route, components = _weighted_rank_route(spec, routes)
        routes[name] = route
        metadata[name] = {
            "name": name,
            "family": str(spec["family"]),
            "kind": "weighted_daily_rank",
            "components": components,
            "audit": _route_audit(route),
        }

    if set(routes) != set(names):
        missing = sorted(set(names).difference(routes))
        raise RuntimeError(f"not all manifest routes were materialized: {missing}")

    batch_dir = output_dir / "route_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for name, route in routes.items():
        path = batch_dir / f"{name}.parquet"
        route.to_parquet(path, index=False)
        metadata[name]["materialized_path"] = str(path.relative_to(root))
        metadata[name]["materialized_sha256"] = file_sha256(path)
    return routes, metadata


def _validate_group(
    group: dict[str, Any],
    routes: dict[str, pd.DataFrame],
) -> None:
    years = tuple(map(int, group["years"]))
    for name in group["routes"]:
        if name not in routes:
            raise KeyError(f"group {group['name']} references unknown route {name}")
        available = set(map(int, routes[name]["date"].dt.year.unique()))
        missing = sorted(set(years).difference(available))
        if missing:
            raise ValueError(
                f"route {name} is missing group {group['name']} years {missing}"
            )


def _frame_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _reference_with_exposure_override(
    reference: Any,
    group: dict[str, Any],
    root: Path,
) -> Any:
    override = group.get("exposure_override")
    if not override:
        return reference
    path = _resolve(str(override["path"]), root)
    if not path.is_file():
        raise FileNotFoundError(f"exposure override is missing: {path}")
    raw = pd.read_parquet(path)
    numeric_columns = tuple(
        map(str, override.get("numeric_columns", OFFICIAL_BARRA_STYLE_COLUMNS))
    )
    categorical_columns = tuple(
        map(str, override.get("categorical_columns", ("industry_level1_code",)))
    )
    selected_columns = (*KEY_COLUMNS, *numeric_columns, *categorical_columns)
    missing = sorted(set(selected_columns).difference(raw.columns))
    if missing:
        raise ValueError(f"exposure override is missing columns: {missing}")
    exposures = raw.loc[:, list(selected_columns)].copy()
    exposures["date"] = pd.to_datetime(exposures["date"], errors="raise").dt.normalize()
    exposures["instrument"] = exposures["instrument"].astype("string")
    years = tuple(map(int, group["years"]))
    exposures = exposures.loc[
        exposures["date"].dt.year.isin(years)
    ].reset_index(drop=True)
    if exposures.empty:
        raise ValueError("exposure override has no rows in the group period")
    if exposures.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("exposure override contains duplicate date-instrument keys")

    rebuilt = type(reference)(
        reference.reference_panel,
        reference.labels,
        exposures,
        reference.reference_columns,
        config=reference.config,
    )
    if hasattr(reference, "candidate454_metadata"):
        rebuilt.candidate454_metadata = reference.candidate454_metadata
    rebuilt.exposure_audit = {
        "profile": str(override.get("profile", "explicit_exposure_override")),
        "evidence_boundary": str(
            override.get(
                "evidence_boundary",
                "Downloaded platform exposure proxy; not a private platform score.",
            )
        ),
        "source_files": [
            {
                "year": year,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for year in years
        ],
        "source_all_columns": list(map(str, raw.columns)),
        "excluded_source_columns": [
            str(column) for column in raw.columns if column not in selected_columns
        ],
    }
    return rebuilt


def _exposure_metadata(
    reference: Any,
    data_dir: Path,
    years: tuple[int, ...],
) -> dict[str, Any]:
    exposures = reference.exposures
    if exposures is None or exposures.empty:
        raise RuntimeError("joint official-period proxy requires non-empty exposures")
    exposure_columns = [column for column in exposures.columns if column not in KEY_COLUMNS]
    numeric_columns = [
        column
        for column in exposure_columns
        if pd.api.types.is_numeric_dtype(exposures[column])
    ]
    dropped_numeric_columns: list[str] = []
    if "SIZE" in numeric_columns and "float_market_cap" in numeric_columns:
        numeric_columns.remove("float_market_cap")
        dropped_numeric_columns.append("float_market_cap")
    categorical_columns = [
        column
        for column in exposure_columns
        if (
            isinstance(exposures[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(exposures[column])
            or pd.api.types.is_string_dtype(exposures[column])
        )
    ]
    exposure_audit = getattr(reference, "exposure_audit", {})
    source_files = exposure_audit.get("source_files")
    if source_files is None:
        source_files = []
        for year in years:
            path = data_dir / f"exposures/year={year}/part-{year}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"exposure source is missing: {path}")
            source_files.append(
                {
                    "year": year,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    ordered_columns = [*KEY_COLUMNS, *exposure_columns]
    per_column = {}
    for column in exposure_columns:
        missing_rows = int(exposures[column].isna().sum())
        per_column[column] = {
            "dtype": str(exposures[column].dtype),
            "nonmissing_rows": int(exposures[column].notna().sum()),
            "missing_rows": missing_rows,
            "missing_ratio": float(missing_rows / len(exposures)),
        }
    return {
        "method": "daily_ols_residual_after_daily_1pct_99pct_winsor_and_zscore",
        "profile": exposure_audit.get("profile", "reduced_available_style_proxy"),
        "evidence_boundary": exposure_audit.get(
            "evidence_boundary",
            "Local available-style proxy, not a complete proprietary BARRA model.",
        ),
        "rows": len(exposures),
        "days": int(exposures["date"].nunique()),
        "instruments": int(exposures["instrument"].nunique()),
        "date_min": str(exposures["date"].min().date()),
        "date_max": str(exposures["date"].max().date()),
        "all_columns": ordered_columns,
        "exposure_columns": exposure_columns,
        "regression_numeric_columns": numeric_columns,
        "regression_categorical_columns": categorical_columns,
        "dropped_redundant_numeric_columns": dropped_numeric_columns,
        "includes_intercept": True,
        "per_column": per_column,
        "panel_sha256": _frame_sha256(exposures.loc[:, ordered_columns]),
        "source_files": source_files,
        "source_all_columns": exposure_audit.get("source_all_columns", ordered_columns),
        "excluded_source_columns": exposure_audit.get("excluded_source_columns", []),
    }


def _summarize_period(
    name: str,
    family: str,
    years: tuple[int, ...],
    score: dict[str, float],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "route": name,
        "family": family,
        "years": list(years),
        "J": float(score["score_proxy"]),
        "A": float(score["a_proxy"]),
        "B": float(score["b_proxy"]),
        # Stable display alias requested for every full J table.
        "score": float(score["score_proxy"]),
    }
    for key in DETAIL_COLUMNS:
        if key in score:
            output[key] = float(score[key])
    return output


def _write_reports(
    output_dir: Path,
    manifest: dict[str, Any],
    route_metadata: dict[str, dict[str, Any]],
    group_results: list[dict[str, Any]],
) -> None:
    protocol = str(manifest.get("protocol", PROTOCOL))
    output = {
        "protocol": protocol,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evidence_boundary": manifest.get(
            "evidence_boundary",
            "Local Candidate454 joint proxy; not a private platform score.",
        ),
        "includes_checkpoint_training_period": bool(
            manifest.get("includes_checkpoint_training_period", False)
            or "in_sample_inclusive" in protocol
        ),
        "note": (
            "Every group is scored once over its complete continuous period in "
            "one shared Elastic Net fit containing Candidate454 and every route "
            "declared in that group. A/B/J are local proxies, not platform scores."
        ),
        "manifest": manifest,
        "routes": route_metadata,
        "groups": group_results,
    }
    (output_dir / "joint_official_proxy_detailed.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_rows = []
    for group in group_results:
        summaries = group["summaries"]
        summary_rows.extend(
            {"group": group["name"], **summary} for summary in summaries
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["group", "J"], ascending=[True, False]
    )
    summary["rank"] = summary.groupby("group", sort=False).cumcount() + 1
    missing_full_columns = [
        column for column in FULL_RESULT_COLUMNS if column not in summary.columns
    ]
    if missing_full_columns:
        raise RuntimeError(
            "joint result is missing required full-detail columns: "
            f"{missing_full_columns}"
        )
    summary.loc[:, SUMMARY_RESULT_COLUMNS].to_csv(
        output_dir / "joint_official_proxy_summary.csv", index=False
    )
    summary.loc[:, FULL_RESULT_COLUMNS].to_csv(
        output_dir / "joint_official_proxy_ab_details.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/candidate454_joint_official_proxy_20260804"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--candidate454-store", type=Path, default=None)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    manifest_path = _resolve(str(args.manifest), root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (root / args.output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    routes, route_metadata = materialize_routes(
        manifest, root, output_dir
    )

    group_results = []
    reference_cache: dict[str, Any] = {}
    for group in manifest["groups"]:
        _validate_group(group, routes)
        years = tuple(map(int, group["years"]))
        reference_key = json.dumps(
            {
                "years": years,
                "exposure_override": group.get("exposure_override"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        reference = reference_cache.get(reference_key)
        if reference is None:
            reference = load_score_reference(
                args.data_dir,
                args.reports_dir,
                years,
                candidate454_store=args.candidate454_store,
            )
            reference = _reference_with_exposure_override(
                reference,
                group,
                root,
            )
            reference_cache[reference_key] = reference
        selected = tuple(map(str, group["routes"]))
        period_routes = {
            name: routes[name].loc[
                routes[name]["date"].dt.year.isin(years)
            ].reset_index(drop=True)
            for name in selected
        }
        period_scores = reference.score_joint_routes(period_routes)
        summaries = [
            _summarize_period(
                name,
                route_metadata[name]["family"],
                years,
                period_scores[name],
            )
            for name in selected
        ]
        summaries.sort(key=lambda row: row["J"], reverse=True)
        group_results.append(
            {
                "name": str(group["name"]),
                "purpose": str(group["purpose"]),
                "years": list(years),
                "route_count": len(selected),
                "reference_factor_count": len(reference.reference_columns),
                "exposure_preprocessing": _exposure_metadata(
                    reference,
                    args.data_dir,
                    years,
                ),
                "period_scores": period_scores,
                "summaries": summaries,
            }
        )
        _write_reports(
            output_dir,
            manifest,
            route_metadata,
            group_results,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "protocol": str(manifest.get("protocol", PROTOCOL)),
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "output_dir": str(output_dir),
                "groups": [
                    {
                        "name": group["name"],
                        "route_count": group["route_count"],
                        "top": group["summaries"][:5],
                    }
                    for group in group_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
