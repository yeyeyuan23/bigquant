from __future__ import annotations

"""E4 frozen-OOS accounting contracts."""

import numpy as np
import pandas as pd
from conftest import FINALS_PRE


def test_e4_launcher_only_builds_labels_infers_and_scores(contract):
    run_source = (FINALS_PRE / "e4_private_fixed_oos/run.sh").read_text(encoding="utf-8")
    infer_source = (FINALS_PRE / "e4_private_fixed_oos/infer_fixed.py").read_text(
        encoding="utf-8"
    )
    assert contract["experiments"]["E4"]["retraining_allowed"] is False
    assert "train" not in run_source.lower()
    assert '"$experiment/infer_fixed.py"' in run_source
    assert '"$experiment/score_fixed.py"' in run_source
    assert "EXPECTED_CHECKPOINT_SHA256" in infer_source
    assert "optimizer" not in infer_source.lower()


def make_day(day: str) -> pd.DataFrame:
    size = 100
    factor = np.arange(size, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.Timestamp(day),
            "instrument": [f"S{index:03d}" for index in range(size)],
            "neutral_factor": factor,
            "ret_next_open_to_close": factor / 100_000.0,
        }
    )


def test_target_weights_are_unit_long_unit_short(load_experiment_module):
    score = load_experiment_module("e4_private_fixed_oos/score_fixed.py", "pre_e4_score")
    weights = score.target_weights(make_day("2025-03-03"))
    assert np.isclose(weights[weights > 0].sum(), 1.0)
    assert np.isclose(weights[weights < 0].sum(), -1.0)
    assert len(weights) == 40


def test_first_entry_from_cash_and_unchanged_portfolio_turnover(load_experiment_module):
    score = load_experiment_module("e4_private_fixed_oos/score_fixed.py", "pre_e4_turnover")
    frame = pd.concat((make_day("2025-03-03"), make_day("2025-03-04")), ignore_index=True)
    daily = score.backtest_daily(frame)
    assert daily["turnover"].tolist() == [1.0, 0.0]


def test_cost_is_one_way_bps_times_target_turnover(load_experiment_module, contract):
    score = load_experiment_module("e4_private_fixed_oos/score_fixed.py", "pre_e4_cost")
    daily = score.backtest_daily(make_day("2025-03-03"))
    row = daily.iloc[0]
    for bps in contract["experiments"]["E4"]["one_way_cost_bps"]:
        if bps == 0:
            continue
        expected = row["gross_return"] - row["turnover"] * bps / 10_000.0
        assert np.isclose(row[f"net_return_{bps}bp"], expected)
