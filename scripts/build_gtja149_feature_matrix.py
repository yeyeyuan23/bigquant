"""Build the 149 retained GTJA candidates from allowed daily bar inputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import polars as pl

KEYS = ("date", "instrument")
SOURCE_PATTERN = re.compile(r"^GTJA-(\d{4})$")
EXTERNAL_COMMIT = "5cf7e83637b85e4f855daec16099148b358b89b3"


def literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def retained_gtja_candidates(candidate_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((candidate_root / "pv").glob("pv_*.py")):
        values = literal_assignments(path)
        candidate_id = values.get("CANDIDATE_ID")
        source_id = values.get("SOURCE_RESEARCH_ID")
        if not isinstance(candidate_id, str) or not isinstance(source_id, str):
            continue
        match = SOURCE_PATTERN.fullmatch(source_id)
        if match is None:
            continue
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_id": source_id,
                "gtja_number": int(match.group(1)),
                "orientation": float(values.get("ORIENTATION", 1.0)),
                "candidate_module": str(path),
            }
        )
    if len(rows) != 149:
        raise RuntimeError(f"expected 149 retained GTJA candidates, found {len(rows)}")
    if len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate retained GTJA candidate ids")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(data_root: Path, years: list[int]) -> pl.DataFrame:
    frames = []
    columns = [
        "date",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
        "volume",
    ]
    for year in years:
        path = data_root / f"features/PV/year={year}/part-{year}.parquet"
        frame = pl.read_parquet(path, columns=columns).with_columns(
            pl.col("date").cast(pl.Datetime),
            pl.col("instrument").cast(pl.String),
            *[
                pl.col(column).cast(pl.Float64)
                for column in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "amount",
                    "volume",
                )
            ],
        )
        frames.append(frame)
    panel = (
        pl.concat(frames, how="vertical")
        .rename({"date": "trade_date", "instrument": "stock_code"})
        .sort(["stock_code", "trade_date"])
    )
    if panel.select(pl.struct(["trade_date", "stock_code"]).is_duplicated().any()).item():
        raise ValueError("daily PV panel contains duplicate date-instrument keys")
    panel = panel.with_columns(
        (pl.col("close") / pl.col("pre_close") - 1.0).alias("returns"),
        (pl.col("amount") / pl.col("volume").replace(0.0, None)).alias("vwap"),
        pl.col("amount")
        .rolling_mean(window_size=20, min_samples=5)
        .over("stock_code")
        .alias("cap"),
    )
    return panel


def normalized_factor(
    panel: pl.DataFrame,
    raw: pl.Series,
    *,
    name: str,
    orientation: float,
) -> tuple[pl.Series, pl.Series, dict[str, object]]:
    scored = panel.select("trade_date").with_columns(raw.alias("__raw"))
    scored = scored.with_columns(
        pl.col("__raw").is_finite().fill_null(False).alias("__available"),
        pl.when(pl.col("__raw").is_finite())
        .then(pl.col("__raw"))
        .otherwise(None)
        .alias("__finite"),
    )
    scored = scored.with_columns(
        pl.col("__finite")
        .fill_null(pl.col("__finite").median().over("trade_date"))
        .alias("__filled")
    )
    scored = scored.with_columns(
        pl.col("__filled").rank(method="average").over("trade_date").alias("__rank"),
        pl.col("__filled").count().over("trade_date").alias("__count"),
    )
    factor = scored.select(
        (
            orientation
            * 2.0
            * (pl.col("__rank") - (pl.col("__count") + 1.0) / 2.0)
            / pl.col("__count")
        )
        .fill_nan(0.0)
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias(name)
    ).to_series()
    available = scored["__available"].cast(pl.Boolean).rename(name)
    stats = {
        "candidate_id": name,
        "available_rows": int(available.sum()),
        "available_fraction": float(available.mean()),
        "finite_output": bool(factor.is_finite().all()),
    }
    return factor, available, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party/aurumq_gtja191",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    parser.add_argument("--end-date", type=str)
    args = parser.parse_args()

    sys.path.insert(0, str(args.external_root))
    from aurumq_rl.factors.gtja191 import REGISTRY

    candidates = retained_gtja_candidates(args.candidate_root)
    panel = load_panel(args.data_root, args.years)
    if args.end_date is not None:
        end_date = pl.lit(args.end_date).str.to_datetime()
        panel = panel.filter(pl.col("trade_date") <= end_date)
        if panel.is_empty():
            raise ValueError("end-date removed every GTJA input row")
    keys = panel.select(
        pl.col("trade_date").alias("date"),
        pl.col("stock_code").alias("instrument"),
    )
    feature_series = []
    availability_series = []
    lineage = []
    for index, row in enumerate(candidates, start=1):
        number = int(row["gtja_number"])
        registry_id = f"gtja_{number:03d}"
        entry = REGISTRY[registry_id]
        raw = entry.impl(panel)
        factor, available, stats = normalized_factor(
            panel,
            raw,
            name=str(row["candidate_id"]),
            orientation=float(row["orientation"]),
        )
        feature_series.append(factor)
        availability_series.append(available)
        lineage.append(
            {
                **row,
                **stats,
                "registry_id": registry_id,
                "quality_flag": int(entry.quality_flag),
                "external_commit": EXTERNAL_COMMIT,
                "cap_proxy": "20-day rolling mean amount derived from bar1m daily aggregation",
            }
        )
        print(
            f"[{index:03d}/{len(candidates)}] {row['candidate_id']} "
            f"source={row['source_id']} coverage={stats['available_fraction']:.6f}",
            flush=True,
        )

    features = keys.with_columns(feature_series).sort(["date", "instrument"])
    availability = keys.with_columns(availability_series).sort(["date", "instrument"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "gtja149_features_wide.parquet"
    availability_path = args.output_dir / "gtja149_availability_wide.parquet"
    lineage_path = args.output_dir / "gtja149_lineage.json"
    features.write_parquet(features_path, compression="zstd")
    availability.write_parquet(availability_path, compression="zstd")
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "gtja-feature-matrix-v2",
        "candidate_count": len(candidates),
        "rows": features.height,
        "date_min": str(features["date"].min().date()),
        "date_max": str(features["date"].max().date()),
        "end_date": args.end_date,
        "factor_sources": ["bar1m"],
        "external_implementation": {
            "repository": "https://github.com/yupoet/aurumq-rl",
            "commit": EXTERNAL_COMMIT,
            "license": "MIT",
        },
        "features": str(features_path),
        "availability": str(availability_path),
        "lineage": str(lineage_path),
        "feature_sha256": sha256(features_path),
        "availability_sha256": sha256(availability_path),
    }
    (args.output_dir / "gtja149_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
