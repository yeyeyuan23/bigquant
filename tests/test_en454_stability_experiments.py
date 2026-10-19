from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_en454_stability_experiments.py"
SPEC = importlib.util.spec_from_file_location("en454_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_continuous_blocks_keep_one_day_label_gap() -> None:
    dates = pd.bdate_range("2020-01-01", periods=105)
    blocks = MODULE.continuous_blocks(dates)
    training, prediction = blocks[0]
    assert len(training) == 60
    assert len(prediction) == 20
    assert training[-1] == prediction[0] - 2


def test_ema_update_uses_fixed_half_retention() -> None:
    first, intercept = MODULE.ema_update(None, None, np.array([1.0, 0.0]), 0.2)
    second, second_intercept = MODULE.ema_update(
        first, intercept, np.array([0.0, 2.0]), 0.6
    )
    np.testing.assert_allclose(second, [0.5, 1.0])
    assert second_intercept == 0.4


def test_clean_screen_drops_low_coverage_and_persistent_reverse_signal() -> None:
    rng = np.random.default_rng(7)
    days, stocks, features = 60, 80, 70
    targets = rng.normal(size=(days, stocks)).astype(np.float32)
    raw = rng.normal(size=(days, stocks, features)).astype(np.float32)
    raw[:, :20, 0] = np.nan
    raw[:, :, 1] = -targets
    raw[:, :, 2] = targets
    zvalues = MODULE.daily_cross_sectional_zscore(raw)
    selected, audit = MODULE.select_clean_features(raw, zvalues, targets)
    assert 0 not in selected
    assert 1 not in selected
    assert 2 in selected
    assert audit["selected_features"] >= MODULE.MIN_SELECTED_FEATURES


def test_clean_coverage_ignores_rows_without_training_targets() -> None:
    rng = np.random.default_rng(9)
    days, stocks, features = 60, 80, 70
    targets = rng.normal(size=(days, stocks)).astype(np.float32)
    targets[:, 60:] = np.nan
    raw = rng.normal(size=(days, stocks, features)).astype(np.float32)
    raw[:, 60:] = np.nan
    raw[:, :, 0] = np.nan
    raw[:, :60, 1] = targets[:, :60]
    zvalues = MODULE.daily_cross_sectional_zscore(raw)
    selected, audit = MODULE.select_clean_features(raw, zvalues, targets)
    assert 0 not in selected
    assert 1 in selected
    assert audit["coverage_denominator_rows"] == days * 60
    assert audit["coverage_pass"] == features - 1


def test_positive_elasticnet_has_no_negative_coefficients() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(8, 60, 6)).astype(np.float32)
    target = values[:, :, 0] - values[:, :, 1]
    columns = np.arange(values.shape[2])
    coefficient, _, _ = MODULE.fit_model(values, target, columns, positive=True)
    assert np.all(coefficient >= -1e-12)
