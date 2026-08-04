from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_unified_temporal.py"
SPEC = importlib.util.spec_from_file_location("evaluate_unified_temporal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
unified_temporal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(unified_temporal)

ELASTICNET_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_unified_elasticnet.py"
)
ELASTICNET_SPEC = importlib.util.spec_from_file_location(
    "evaluate_unified_elasticnet",
    ELASTICNET_SCRIPT,
)
assert ELASTICNET_SPEC is not None and ELASTICNET_SPEC.loader is not None
unified_elasticnet = importlib.util.module_from_spec(ELASTICNET_SPEC)
ELASTICNET_SPEC.loader.exec_module(unified_elasticnet)

MERGE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_candidate454_elasticnet_oos.py"
)
MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_candidate454_elasticnet_oos",
    MERGE_SCRIPT,
)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
merge_elasticnet = importlib.util.module_from_spec(MERGE_SPEC)
MERGE_SPEC.loader.exec_module(merge_elasticnet)


@pytest.mark.parametrize(
    ("half", "expected"),
    [
        (
            "H1",
            (
                pd.Timestamp("2022-12-31"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-06-30"),
            ),
        ),
        (
            "h2",
            (
                pd.Timestamp("2023-06-30"),
                pd.Timestamp("2023-07-01"),
                pd.Timestamp("2023-12-31"),
            ),
        ),
    ],
)
def test_fold_boundaries_are_causal_and_cover_each_half(
    half: str,
    expected: tuple[pd.Timestamp, ...],
) -> None:
    assert unified_temporal.fold_boundaries(2023, half) == expected


def test_fold_boundaries_reject_unknown_half() -> None:
    with pytest.raises(ValueError, match="unsupported validation half"):
        unified_temporal.fold_boundaries(2023, "q1")


def test_eligible_target_indices_drop_empty_and_single_stock_dates() -> None:
    targets = np.array(
        [
            [np.nan, np.nan, np.nan],
            [0.1, np.nan, np.nan],
            [0.1, -0.2, np.nan],
            [0.1, -0.2, 0.3],
        ]
    )
    eligible, skipped = unified_temporal.eligible_target_indices(targets, [0, 1, 2, 3])
    assert eligible == [2, 3]
    assert skipped == 2


def test_eligible_label_dates_exclude_unscorable_boundary_dates() -> None:
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-12-30", "2024-12-30", "2024-12-31", "2024-12-31"]
            ),
            "ret_next_open_to_close": [0.01, -0.02, np.nan, np.nan],
        }
    )

    dates = unified_temporal.eligible_label_dates(labels)

    assert dates.tolist() == [pd.Timestamp("2024-12-30")]


def test_residualize_targets_removes_daily_baseline_projection() -> None:
    targets = np.array(
        [
            [-1.0, 0.3, 1.0],
            [0.2, np.nan, -0.2],
        ],
        dtype=np.float32,
    )
    baseline = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    residuals, betas = unified_temporal.residualize_targets_against_baseline(
        targets,
        baseline,
    )

    valid = np.isfinite(residuals[0])
    centered_baseline = baseline[0, valid] - baseline[0, valid].mean()
    assert abs(float(np.dot(residuals[0, valid], centered_baseline))) < 1e-6
    assert abs(float(residuals[0, valid].mean())) < 1e-6
    assert np.isfinite(betas).all()


def test_incremental_loss_rewards_residual_signal_and_penalizes_baseline_copy() -> None:
    torch = pytest.importorskip("torch")
    baseline = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    residual = torch.tensor([1.0, -1.0, -1.0, 1.0])

    good_loss, good_diagnostics = unified_temporal.incremental_correlation_loss(
        residual,
        residual,
        baseline,
        orthogonality_weight=1.0,
        stability_weight=0.3,
        correlation_floor=0.0,
    )
    copied_loss, copied_diagnostics = unified_temporal.incremental_correlation_loss(
        baseline,
        residual,
        baseline,
        orthogonality_weight=1.0,
        stability_weight=0.3,
        correlation_floor=0.0,
    )

    assert good_loss < copied_loss
    assert good_diagnostics["residual_correlation"] > 0.99
    assert abs(float(good_diagnostics["baseline_correlation"])) < 1e-6
    assert copied_diagnostics["baseline_correlation"] > 0.99


def test_incremental_loss_stability_term_penalizes_negative_incremental_ic() -> None:
    torch = pytest.importorskip("torch")
    baseline = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    residual = torch.tensor([1.0, -1.0, -1.0, 1.0])

    _, diagnostics = unified_temporal.incremental_correlation_loss(
        -residual,
        residual,
        baseline,
        orthogonality_weight=0.0,
        stability_weight=0.3,
        correlation_floor=0.0,
    )

    assert diagnostics["residual_correlation"] < -0.99
    assert diagnostics["downside_penalty"] > 0.99


def test_load_aligned_baseline_route_preserves_missing_keys_and_lineage(
    tmp_path: Path,
) -> None:
    route_path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "instrument": ["A", "B", "A"],
            "factor": [-0.5, 0.5, 0.2],
        }
    ).to_parquet(route_path, index=False)

    aligned, metadata = unified_temporal.load_aligned_baseline_route(
        route_path,
        pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        ("A", "B"),
    )

    np.testing.assert_allclose(aligned[0], [-0.5, 0.5])
    assert aligned[1, 0] == pytest.approx(0.2)
    assert np.isnan(aligned[1, 1])
    assert metadata["matched_rows"] == 3
    assert metadata["missing_rows"] == 1
    assert len(metadata["sha256"]) == 64


def test_residual_baseline_manifest_requires_strict_oos_year_coverage(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        """{
  "model": "elastic_net",
  "role": "candidate454_full_pool_oos_baseline",
  "feature_bundle": "candidate454",
  "candidate_feature_count": 454,
  "label_isolation_gap_days": 1,
  "training_protocol": "60d_train_20d_predict",
  "alpha": 0.001,
  "l1_ratio": 0.5,
  "neutral_filled_rows": 0,
  "continuous_oos": true,
  "years": [2019, 2020, 2021, 2022, 2023]
}
""",
        encoding="utf-8",
    )

    metadata = unified_temporal.validate_residual_baseline_manifest(
        manifest_path,
        required_years={2019, 2020, 2021, 2022, 2023},
        expected_candidate_count=454,
    )

    assert metadata["payload"]["candidate_feature_count"] == 454
    with pytest.raises(ValueError, match="years"):
        unified_temporal.validate_residual_baseline_manifest(
            manifest_path,
            required_years={2019, 2020, 2021, 2022, 2023, 2024},
            expected_candidate_count=454,
        )


def test_load_aligned_baseline_route_excludes_constant_neutral_fill_days(
    tmp_path: Path,
) -> None:
    route_path = tmp_path / "baseline.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "instrument": ["A", "B", "A", "B"],
            "factor": [0.0, 0.0, -0.4, 0.4],
        }
    ).to_parquet(route_path, index=False)

    aligned, metadata = unified_temporal.load_aligned_baseline_route(
        route_path,
        pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        ("A", "B"),
    )

    assert np.isnan(aligned[0]).all()
    np.testing.assert_allclose(aligned[1], [-0.4, 0.4])
    assert metadata["excluded_constant_days"] == 1
    assert metadata["excluded_constant_rows"] == 2


def test_elasticnet_blocks_use_60_days_with_one_day_label_gap() -> None:
    dates = pd.date_range("2022-09-01", "2023-03-31", freq="B")
    blocks = unified_elasticnet.rolling_blocks(
        dates,
        (2023,),
        train_days=60,
        prediction_days=20,
    )
    training, prediction = blocks[0]
    assert len(training) == 60
    assert len(prediction) == 20
    assert training[-1] == prediction[0] - 2
    assert dates[prediction].year.nunique() == 1


def test_elasticnet_continuous_blocks_start_after_warmup_and_label_gap() -> None:
    dates = pd.date_range("2019-01-02", periods=108, freq="B")
    blocks = unified_elasticnet.continuous_rolling_blocks(
        dates,
        train_days=60,
        prediction_days=20,
        label_gap_days=1,
    )

    first_training, first_prediction = blocks[0]
    last_training, last_prediction = blocks[-1]
    assert first_training.tolist() == list(range(60))
    assert first_prediction.tolist() == list(range(61, 81))
    assert first_training[-1] == first_prediction[0] - 2
    assert last_training[-1] == last_prediction[0] - 2
    assert last_prediction[-1] == len(dates) - 1


def test_elasticnet_daily_zscore_masks_missing_and_constant_features() -> None:
    values = np.array([[[1.0, 5.0], [3.0, 5.0], [np.nan, 5.0]]])
    normalized = unified_elasticnet.daily_cross_sectional_zscore(values)
    np.testing.assert_allclose(normalized[0, :, 0], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(normalized[0, :, 1], [0.0, 0.0, 0.0])
    assert np.isfinite(normalized).all()


def test_merge_candidate454_baseline_partitions_preserves_oos_lineage(
    tmp_path: Path,
) -> None:
    partitions = []
    for year in (2022, 2023):
        directory = tmp_path / str(year)
        directory.mkdir()
        pd.DataFrame(
            {
                "date": pd.to_datetime([f"{year}-01-03", f"{year}-01-03"]),
                "instrument": ["A", "B"],
                "factor": [-0.5, 0.5],
            }
        ).to_parquet(
            directory / "candidate454_elasticnet_full_oos.parquet",
            index=False,
        )
        (directory / "run_manifest.json").write_text(
            f"""{{
  "model": "elastic_net",
  "role": "candidate454_full_pool_oos_baseline",
  "feature_bundle": "candidate454",
  "candidate_feature_count": 454,
  "training_protocol": "60d_train_20d_predict",
  "label_isolation_gap_days": 1,
  "years": [{year}],
  "alpha": 0.001,
  "l1_ratio": 0.5,
  "train_start_year": 2019,
  "neutral_filled_rows": 0
}}
""",
            encoding="utf-8",
        )
        partitions.append(directory)

    route, manifest = merge_elasticnet.merge_partitions(partitions)

    assert len(route) == 4
    assert manifest["years"] == [2022, 2023]
    assert manifest["continuous_oos"] is True
    assert manifest["partitioned_by_year"] is True
    assert len(manifest["partitions"]) == 2
