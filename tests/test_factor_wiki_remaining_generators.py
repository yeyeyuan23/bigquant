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

HAITONG_GENERATOR_PATH = GENERATOR_ROOT / "build_haitong_components.py"
HAITONG_SPEC = importlib.util.spec_from_file_location(
    "build_haitong_components", HAITONG_GENERATOR_PATH
)
assert HAITONG_SPEC is not None and HAITONG_SPEC.loader is not None
haitong = importlib.util.module_from_spec(HAITONG_SPEC)
HAITONG_SPEC.loader.exec_module(haitong)


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


def test_exact_remaining_132_manifest_split() -> None:
    with (
        GENERATOR_ROOT / "submission_manifest_changjiang_123.csv"
    ).open(encoding="utf-8", newline="") as handle:
        changjiang_rows = list(csv.DictReader(handle))
    with (GENERATOR_ROOT / "submission_manifest_haitong_11.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        haitong_rows = list(csv.DictReader(handle))

    changjiang_hf = [
        row for row in changjiang_rows if row["family"] == "HF"
    ]
    changjiang_pv = [
        row for row in changjiang_rows if row["candidate_id"] == "PV-219"
    ]
    haitong_hf = [row for row in haitong_rows if row["family"] == "HF"]
    remaining = changjiang_hf + changjiang_pv + haitong_hf

    assert len(changjiang_hf) == 120
    assert len(changjiang_pv) == 1
    assert len(haitong_hf) == 11
    assert len(remaining) == 132
    assert len({row["source_id"] for row in remaining}) == 132
    assert len({row["candidate_id"] for row in remaining}) == 132
    assert all(row["semantic_class"] == "LATENT_COMPONENT" for row in remaining)
    assert all(row["include_in_j_baseline"] == "True" for row in remaining)
    assert changjiang_pv[0]["source_id"] == "CJ-G112-V01"


def test_haitong_factor_panel_emits_all_12_remaining_components() -> None:
    dates = pd.bdate_range("2020-01-02", periods=30)
    base = pd.DataFrame(
        {
            "date": dates,
            "instrument": "A",
            "open": np.linspace(10.0, 11.0, 30),
            "high": np.linspace(10.2, 11.2, 30),
            "low": np.linspace(9.8, 10.8, 30),
            "close": np.linspace(10.1, 11.1, 30),
            "vwap": np.linspace(10.05, 11.05, 30),
            "tail60_amount": np.linspace(100.0, 200.0, 30),
            "total_amount": np.linspace(500.0, 700.0, 30),
            "l1_strength_daily": np.linspace(-0.1, 0.1, 30),
        }
    )
    # These are the minute primitives consumed by the frozen formula mapping.
    for column in (
        "m1_rv_origin",
        "m1_rv_central",
        "m5_rv_origin",
        "m5_rv_central",
        "m5_all_offsets_rv_central",
        "m1_skew_origin",
        "m1_skew_central",
        "m5_skew_origin",
        "m5_skew_central",
        "m5_all_offsets_skew_central",
        "m1_kurt_origin",
        "m1_kurt_central",
        "m5_kurt_origin",
        "m5_kurt_central",
        "m5_all_offsets_kurt_central",
        "m1_up_vol",
        "m1_down_vol",
        "m1_up_ratio",
        "m1_down_ratio",
        "m5_up_vol",
        "m5_down_vol",
        "m5_up_ratio",
        "m5_down_ratio",
        "m10_up_vol",
        "m10_down_vol",
        "m10_up_ratio",
        "m10_down_ratio",
    ):
        base[column] = np.linspace(1.0, 2.0, 30)

    panel, formulas = haitong.build_factor_panel(base)
    with (GENERATOR_ROOT / "submission_manifest_haitong_11.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        expected = {row["component_column"] for row in csv.DictReader(handle)}

    assert expected.issubset(panel.columns)
    assert expected.issubset(formulas)


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
