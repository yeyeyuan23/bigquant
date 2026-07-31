from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "scripts" / "factor_wiki_remaining"
GENERATOR_PATH = GENERATOR_ROOT / "build_changjiang_components.py"
SPEC = importlib.util.spec_from_file_location(
    "build_changjiang_components", GENERATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
changjiang = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(changjiang)


def test_changjiang_manifest_maps_all_123_submitted_components() -> None:
    manifest_path = GENERATOR_ROOT / "submission_manifest_changjiang_123.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 123
    assert len({row["source_id"] for row in rows}) == 123
    assert len({row["candidate_id"] for row in rows}) == 123
    assert all(row["component_column"] == row["source_id"] for row in rows)
    assert all(not Path(row["target_path"]).is_absolute() for row in rows)


def test_cicc_manifest_maps_all_13_submitted_components() -> None:
    manifest_path = GENERATOR_ROOT / "submission_manifest_cicc_13.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 13
    assert len({row["source_id"] for row in rows}) == 13
    assert len({row["candidate_id"] for row in rows}) == 13
    assert all(row["component_column"] == row["source_id"] for row in rows)
    assert all(not Path(row["target_path"]).is_absolute() for row in rows)


def test_rolling_spearman_is_causal_under_future_perturbation() -> None:
    dates = pd.bdate_range("2020-01-02", periods=30)
    base = pd.DataFrame(
        {
            "date": dates,
            "instrument": "A",
            "volume": np.arange(1.0, 31.0),
            "close": np.linspace(10.0, 13.0, 30),
        }
    )
    original = changjiang._rolling_spearman(
        base, base["volume"], base["close"], 21
    )
    changed = base.copy()
    changed.loc[25:, "volume"] = [1000.0, 5.0, 900.0, 4.0, 800.0]
    perturbed = changjiang._rolling_spearman(
        changed, changed["volume"], changed["close"], 21
    )

    pd.testing.assert_series_equal(original.iloc[:25], perturbed.iloc[:25])


def test_method_one_is_pooled_cv_not_mean_daily_cv() -> None:
    base = pd.DataFrame(
        {
            "instrument": "A",
            "m5_volume_n": [2.0] * 20,
            "m5_volume_sum": [3.0] * 20,
            "m5_volume_sum2": [5.0] * 20,
        }
    )
    result = changjiang._rolling_pooled_cv(base, "m5", "volume", 20)

    pooled_n = 40.0
    pooled_sum = 60.0
    pooled_sum2 = 100.0
    variance = (pooled_sum2 - pooled_sum**2 / pooled_n) / (pooled_n - 1.0)
    expected = np.sqrt(variance) / (pooled_sum / pooled_n)
    assert np.isclose(result.iloc[-1], expected)
