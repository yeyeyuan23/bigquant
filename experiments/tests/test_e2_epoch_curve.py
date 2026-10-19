from __future__ import annotations

"""E2 epoch-curve contracts."""

import numpy as np
import pandas as pd


def test_run_script_scores_all_six_epochs_on_one_declared_oos(contract):
    expected = contract["experiments"]["E2"]
    assert expected["epochs"] == [1, 2, 3, 4, 5, 6]
    assert expected["same_oos_for_every_epoch"] is True


def test_daily_ic_has_the_expected_direction(load_experiment_module):
    score = load_experiment_module("e2_o2c_epoch_curve/score.py", "pre_e2_score")
    rows = []
    for day in pd.date_range("2024-01-02", periods=3, freq="B"):
        factor = np.arange(100, dtype=float)
        for index, value in enumerate(factor):
            rows.append(
                {
                    "date": day,
                    "instrument": f"S{index:03d}",
                    "neutral_factor": value,
                    score.LABEL: value,
                }
            )
    ic = score.daily_ic(pd.DataFrame(rows))
    np.testing.assert_allclose(ic.to_numpy(), np.ones(3))
