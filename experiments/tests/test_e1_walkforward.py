from __future__ import annotations

"""E1 walk-forward contracts."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def e1(load_experiment_module):
    pytest.importorskip("torch")
    return load_experiment_module("e1_o2c_walkforward/train.py", "pre_e1_train")


def test_schedule_matches_the_declared_60_20_contract(e1, contract):
    dates = pd.bdate_range("2019-01-02", "2026-08-28")
    blocks = e1.private_schedule(dates)
    expected = contract["experiments"]["E1"]
    private_count = int(((dates >= e1.PRIVATE_START) & (dates <= e1.PRIVATE_END)).sum())
    expected_blocks = sum(
        private_count - offset >= expected["prediction_days"]
        for offset in range(0, private_count, expected["retrain_step_days"])
    )
    assert len(blocks) == expected_blocks
    assert all(len(prediction) == expected["prediction_days"] for _, prediction in blocks)


def test_every_block_has_one_complete_label_isolation_day(e1):
    dates = pd.bdate_range("2019-01-02", "2026-08-28")
    for training, prediction in e1.private_schedule(dates):
        assert int(training[-1]) == int(prediction[0]) - 2
        assert np.intersect1d(training, prediction).size == 0


def test_adding_future_dates_does_not_change_existing_blocks(e1):
    original = pd.bdate_range("2019-01-02", "2026-08-28")
    extended = pd.bdate_range("2019-01-02", "2027-12-31")
    original_blocks = e1.private_schedule(original)
    extended_blocks = e1.private_schedule(extended)
    assert len(original_blocks) == len(extended_blocks)
    for (train_a, pred_a), (train_b, pred_b) in zip(original_blocks, extended_blocks):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(pred_a, pred_b)


def test_short_private_calendar_is_rejected(e1):
    dates = pd.bdate_range("2019-01-02", periods=1500)
    with pytest.raises(RuntimeError, match="unexpectedly short"):
        e1.private_schedule(dates[dates < "2025-03-01"])
