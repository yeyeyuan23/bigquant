from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.hf.hf_003 import (
    build_hf_003_factor_from_daily,
    compute_hf_003_daily,
)
from bigalpha2026.candidates.hf.hf_004 import (
    build_hf_004_factor_from_daily,
    compute_hf_004_daily,
)
from bigalpha2026.candidates.ob.ob_004 import (
    build_ob_004_factor_from_daily,
    compute_ob_004_daily,
)


def _signed_variation_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=8)
    downside_share = {"A": 0.10, "B": 0.90, "C": 0.50}
    rows = [
        {
            "date": date,
            "instrument": instrument,
            "realized_volatility": 1.0,
            "downside_realized_volatility": np.sqrt(share),
        }
        for date in dates
        for instrument, share in downside_share.items()
    ]
    daily = pd.DataFrame(rows)
    return daily, daily[["date", "instrument"]].copy()


class LiteratureRound3Hf003Test(unittest.TestCase):
    def test_direction_interface_and_future_safety(self) -> None:
        daily, pool = _signed_variation_inputs()
        components = compute_hf_003_daily(daily)
        latest = components.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(latest.loc["B", "factor_raw"], 0.0)
        self.assertLess(latest.loc["A", "factor_raw"], 0.0)

        result = build_hf_003_factor_from_daily(daily, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())

        cutoff = pd.Timestamp("2024-01-09")
        changed = daily.copy()
        changed.loc[
            changed["date"] > cutoff,
            "downside_realized_volatility",
        ] = 0.01
        rerun = build_hf_003_factor_from_daily(changed, pool)
        pd.testing.assert_frame_equal(
            result.loc[result["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


def _residual_pressure_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows: list[dict[str, object]] = []
    for instrument, final_residual in (("A", 0.40), ("B", -0.40), ("C", 0.0)):
        for index, date in enumerate(dates):
            tail_return = 0.01 if index % 2 == 0 else -0.01
            imbalance = 2.0 * tail_return
            if index == len(dates) - 1:
                tail_return = 0.0
                imbalance = final_residual
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "tail_60_log_return": tail_return,
                    "tail_60_volume": 100.0,
                    "tail_60_signed_volume_bvc": 100.0 * imbalance,
                }
            )
    daily = pd.DataFrame(rows)
    return daily, daily[["date", "instrument"]].copy()


class LiteratureRound3Hf004Test(unittest.TestCase):
    def test_residual_direction_interface_and_future_safety(self) -> None:
        daily, pool = _residual_pressure_inputs()
        components = compute_hf_004_daily(
            daily,
            regression_window_days=35,
            regression_min_periods=30,
        )
        latest = components.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(latest.loc["A", "factor_raw"], 0.0)
        self.assertLess(latest.loc["B", "factor_raw"], 0.0)

        result = build_hf_004_factor_from_daily(
            daily,
            pool,
            regression_window_days=35,
            regression_min_periods=30,
        )
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())
        latest_factor = result.loc[result["date"].eq(result["date"].max())].set_index(
            "instrument"
        )
        self.assertGreater(
            latest_factor.loc["A", "factor"],
            latest_factor.loc["B", "factor"],
        )

        cutoff = pd.Timestamp("2024-02-20")
        changed = daily.copy()
        changed.loc[
            changed["date"] > cutoff,
            "tail_60_signed_volume_bvc",
        ] *= -1.0
        rerun = build_hf_004_factor_from_daily(
            changed,
            pool,
            regression_window_days=35,
            regression_min_periods=30,
        )
        pd.testing.assert_frame_equal(
            result.loc[result["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


class LiteratureRound3Ob004Test(unittest.TestCase):
    def test_innovation_direction_and_interface(self) -> None:
        date = pd.Timestamp("2024-03-01")
        daily = pd.DataFrame(
            {
                "date": [date] * 3,
                "instrument": ["A", "B", "C"],
                "full_day_depth_imbalance_median": [0.0, 0.0, 0.0],
                "full_day_depth_imbalance_std": [0.2, 0.2, 0.2],
                "tail_60_bid_depth_imbalance_median": [0.8, -0.8, 0.0],
            }
        )
        pool = daily[["date", "instrument"]].copy()
        components = compute_ob_004_daily(daily).set_index("instrument")
        self.assertGreater(components.loc["A", "factor_raw"], 0.0)
        self.assertLess(components.loc["B", "factor_raw"], 0.0)

        result = build_ob_004_factor_from_daily(daily, pool).set_index("instrument")
        self.assertGreater(result.loc["A", "factor"], result.loc["B", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())


if __name__ == "__main__":
    unittest.main()
