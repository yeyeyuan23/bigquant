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


def test_elasticnet_daily_zscore_masks_missing_and_constant_features() -> None:
    values = np.array([[[1.0, 5.0], [3.0, 5.0], [np.nan, 5.0]]])
    normalized = unified_elasticnet.daily_cross_sectional_zscore(values)
    np.testing.assert_allclose(normalized[0, :, 0], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(normalized[0, :, 1], [0.0, 0.0, 0.0])
    assert np.isfinite(normalized).all()
