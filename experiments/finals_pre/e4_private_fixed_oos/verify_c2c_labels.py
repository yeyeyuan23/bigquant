"""Verify the frozen-model C2C label artifact against adjusted daily closes."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    output_dir = parse_args().output_dir.resolve()
    factor_path = output_dir / "m_raw_frozen_private_oos.parquet"
    labels_path = output_dir / "platform_labels_private.parquet"
    prices_path = output_dir / "adjusted_daily_prices.parquet"

    factor = normalize_keys(pd.read_parquet(factor_path, columns=["date", "instrument"]))
    labels = normalize_keys(pd.read_parquet(labels_path))
    prices = normalize_keys(pd.read_parquet(prices_path))

    expected_label_columns = ["date", "instrument", "ret_close_to_close"]
    if labels.columns.tolist() != expected_label_columns:
        raise RuntimeError(
            f"expected C2C-only columns {expected_label_columns}, found {labels.columns.tolist()}"
        )
    for name, frame in (("factor", factor), ("labels", labels), ("prices", prices)):
        if frame.duplicated(["date", "instrument"]).any():
            raise RuntimeError(f"duplicate {name} keys")

    calendar = sorted(pd.DatetimeIndex(prices["date"].unique()))
    next_day = dict(pairwise(calendar))
    recomputed = factor.copy()
    recomputed["label_date"] = recomputed["date"].map(next_day)
    current_prices = prices.rename(columns={"adjusted_close": "adjusted_close_0"})
    next_prices = prices.rename(
        columns={"date": "label_date", "adjusted_close": "adjusted_close_1"}
    )
    recomputed = recomputed.merge(
        current_prices,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    ).merge(
        next_prices,
        on=["label_date", "instrument"],
        how="left",
        validate="many_to_one",
    )
    recomputed["recomputed_c2c"] = (
        recomputed["adjusted_close_1"] / recomputed["adjusted_close_0"] - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    compared = labels.merge(
        recomputed[["date", "instrument", "recomputed_c2c"]],
        on=["date", "instrument"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise RuntimeError(f"label/recomputed key mismatch: {compared['_merge'].value_counts().to_dict()}")

    stored = compared["ret_close_to_close"].to_numpy(dtype=float)
    expected = compared["recomputed_c2c"].to_numpy(dtype=float)
    stored_null = np.isnan(stored)
    expected_null = np.isnan(expected)
    null_masks_equal = bool(np.array_equal(stored_null, expected_null))
    jointly_finite = np.isfinite(stored) & np.isfinite(expected)
    errors = np.abs(stored[jointly_finite] - expected[jointly_finite])
    differing = int(np.count_nonzero(errors > 1e-12))
    if not null_masks_equal or differing:
        raise RuntimeError(
            f"C2C mismatch: null_masks_equal={null_masks_equal} differing={differing}"
        )

    audit = {
        "label": "ret_close_to_close",
        "formula": "adjusted_close(next trading day) / adjusted_close(current day) - 1",
        "label_columns": labels.columns.tolist(),
        "factor_rows": len(factor),
        "factor_days": int(factor["date"].nunique()),
        "factor_start": factor["date"].min().date().isoformat(),
        "factor_end": factor["date"].max().date().isoformat(),
        "label_rows": len(labels),
        "label_non_null_rows": int(labels["ret_close_to_close"].notna().sum()),
        "label_non_null_days": int(
            labels.loc[labels["ret_close_to_close"].notna(), "date"].nunique()
        ),
        "price_rows": len(prices),
        "price_days": int(prices["date"].nunique()),
        "jointly_finite_rows": int(jointly_finite.sum()),
        "null_masks_equal": null_masks_equal,
        "different_rows_gt_1e_12": differing,
        "max_abs_error": float(errors.max(initial=0.0)),
        "mean_abs_error": float(errors.mean()) if len(errors) else 0.0,
        "label_file_sha256": sha256(labels_path),
    }
    audit_path = output_dir / "c2c_label_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
