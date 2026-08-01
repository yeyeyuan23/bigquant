from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bigalpha2026.candidates.hf.hf_104 import compute_hf_104_daily
from bigalpha2026.candidates.ob.ob_008 import compute_ob_008_daily
from scripts.audit_candidate_source_provenance import audit_provenance
from scripts.factor_wiki_remaining.build_remaining132_feature_matrix import (
    _is_remaining_132,
)
from scripts.factor_wiki_remaining.io_utils import (
    discover_month_paths,
    read_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = REPO_ROOT / "scripts" / "factor_wiki_remaining"
GENERATOR_PATH = GENERATOR_ROOT / "build_changjiang_components.py"
SPEC = importlib.util.spec_from_file_location("build_changjiang_components", GENERATOR_PATH)
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


@pytest.mark.parametrize("suffix", ["parquet", "feather"])
def test_month_input_helpers_support_both_formats(tmp_path: Path, suffix: str) -> None:
    frame = pd.DataFrame({"date": ["2020-01-02"], "close": [123]})
    for month in range(1, 13):
        path = tmp_path / f"2020{month:02d}.0.{suffix}"
        if suffix == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_feather(path)

    paths = discover_month_paths(tmp_path, (2020,))
    assert len(paths) == 12
    assert paths[0].name == f"202001.0.{suffix}"
    pd.testing.assert_frame_equal(read_columns(paths[0], ("close",)), frame[["close"]])


def test_month_input_helpers_reject_duplicate_formats(tmp_path: Path) -> None:
    frame = pd.DataFrame({"close": [123]})
    for month in range(1, 13):
        frame.to_parquet(tmp_path / f"2020{month:02d}.0.parquet", index=False)
    frame.to_feather(tmp_path / "202001.0.feather")

    with pytest.raises(ValueError, match="duplicate monthly inputs for 202001"):
        discover_month_paths(tmp_path, (2020,))


def test_project_components_are_executable_from_raw_daily_primitives() -> None:
    keys = {
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "instrument": ["A", "A"],
    }
    hf = pd.DataFrame(
        {
            **keys,
            "tail_60_signed_volume_bvc": [20.0, -15.0],
            "tail_60_volume": [100.0, 50.0],
            "directional_efficiency": [0.25, 2.0],
        }
    )
    ob = pd.DataFrame(
        {
            **keys,
            "tail_60_relative_spread_median": [0.03, 0.01],
            "full_day_relative_spread_median": [0.02, 0.015],
        }
    )

    hf_result = compute_hf_104_daily(hf)
    ob_result = compute_ob_008_daily(ob)

    np.testing.assert_allclose(hf_result["factor_raw"], [0.15, 0.0])
    np.testing.assert_allclose(ob_result["factor_raw"], [0.01, -0.005])


def test_all_462_candidates_have_source_provenance_classification() -> None:
    rows = audit_provenance()

    assert len(rows) == 462
    assert len({row["candidate_id"] for row in rows}) == 462
    assert not any(str(row["provenance_status"]).startswith("unresolved") for row in rows)
    assert sum(row["evidence_level"] == "formula_only" for row in rows) == 157
    assert (
        sum(
            row["evidence_level"] not in {"executable_candidate", "executable_upstream_generator"}
            for row in rows
        )
        == 157
    )
    assert all(
        set(row["traced_sources"]).issubset({"bar1m", "financial", "instruments"}) for row in rows
    )


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
    with (GENERATOR_ROOT / "submission_manifest_changjiang_123.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        changjiang_rows = list(csv.DictReader(handle))
    with (GENERATOR_ROOT / "submission_manifest_haitong_11.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        haitong_rows = list(csv.DictReader(handle))

    changjiang_hf = [row for row in changjiang_rows if row["family"] == "HF"]
    changjiang_pv = [row for row in changjiang_rows if row["candidate_id"] == "PV-219"]
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

    all_rows = [*changjiang_rows, *haitong_rows]
    remaining_ids = {
        row["candidate_id"] for row in all_rows if _is_remaining_132(row["candidate_id"])
    }
    assert len(remaining_ids) == 132


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
    original = changjiang._rolling_spearman(base, base["volume"], base["close"], 21)
    changed = base.copy()
    changed.loc[25:, "volume"] = [1000.0, 5.0, 900.0, 4.0, 800.0]
    perturbed = changjiang._rolling_spearman(changed, changed["volume"], changed["close"], 21)

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
