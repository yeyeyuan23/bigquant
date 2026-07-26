from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_003 import (
    build_fr_003_factor,
    compute_fr_003_events,
)
from bigalpha2026.candidates.fr.fr_004 import (
    build_fr_004_factor,
    compute_fr_004_events,
)
from bigalpha2026.candidates.fr.fr_005 import build_fr_005_factor
from bigalpha2026.candidates.pv.pv_003 import (
    build_pv_003_factor,
    compute_pv_003_daily,
)
from bigalpha2026.candidates.pv.pv_004 import (
    build_pv_004_factor,
    compute_pv_004_daily,
)
from bigalpha2026.candidates.pv.pv_005 import (
    build_pv_005_factor,
    compute_pv_005_daily,
)
from bigalpha2026.candidates.pv.pv_006 import (
    build_pv_006_factor,
    compute_pv_006_daily,
)
from bigalpha2026.candidates.pv.pv_007 import (
    build_pv_007_factor,
    compute_pv_007_daily,
)


INSTRUMENTS = ("A", "B", "C")


def _pv_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=25)
    return_paths = {
        "A": [0.0] * 20 + [0.20] + [0.0] * 4,
        "B": [0.0] * 20 + [-0.20] + [0.0] * 4,
        "C": [0.01, -0.01] * 12 + [0.01],
    }
    rows: list[dict[str, object]] = []
    for instrument in INSTRUMENTS:
        for index, date in enumerate(dates):
            daily_return = return_paths[instrument][index]
            pre_close = 10.0
            close = pre_close * (1.0 + daily_return)
            range_scale = {"A": 0.08, "B": 0.03, "C": 0.01}[instrument]
            zero_trade = instrument == "A" and index % 2 == 0
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "open": pre_close,
                    "high": pre_close * (1.0 + range_scale),
                    "low": pre_close * (1.0 - range_scale),
                    "close": close,
                    "pre_close": pre_close,
                    "amount": {"A": 1_000.0, "B": 10_000.0, "C": 5_000.0}[
                        instrument
                    ],
                    "volume": 0.0 if zero_trade else 1_000.0,
                    "deal_number": 0.0 if zero_trade else 100.0,
                }
            )
    bars = pd.DataFrame(rows)
    pool = bars[["date", "instrument"]].copy()
    return bars, pool


def _financial_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_dates = (pd.Timestamp("2022-03-31"), pd.Timestamp("2023-03-31"))
    disclosure_dates = (pd.Timestamp("2022-04-08"), pd.Timestamp("2023-04-10"))
    effective_dates = (pd.Timestamp("2022-04-11"), pd.Timestamp("2023-04-11"))
    asset_paths = {
        "A": (100.0, 150.0),
        "B": (100.0, 110.0),
        "C": (100.0, 90.0),
    }
    revenue_paths = {
        "A": (100.0, 140.0),
        "B": (100.0, 110.0),
        "C": (100.0, 90.0),
    }
    cash_flows = {"A": 90.0, "B": 50.0, "C": 10.0}
    rows: list[dict[str, object]] = []
    for instrument in INSTRUMENTS:
        for index, report_date in enumerate(report_dates):
            common = {
                "disclosure_date": disclosure_dates[index],
                "effective_date": effective_dates[index],
                "instrument": instrument,
                "report_date": report_date,
                "shift": 0,
            }
            rows.append(
                {
                    **common,
                    "category": "lf",
                    "net_cffoa": np.nan,
                    "net_profit": np.nan,
                    "operating_revenue": np.nan,
                    "total_assets": asset_paths[instrument][index],
                }
            )
            rows.append(
                {
                    **common,
                    "category": "ttm",
                    "net_cffoa": (
                        cash_flows[instrument] * (0.8 if index == 0 else 1.0)
                    ),
                    "net_profit": 20.0,
                    "operating_revenue": revenue_paths[instrument][index],
                    "total_assets": np.nan,
                }
            )
    pool_dates = pd.bdate_range("2023-04-11", periods=3)
    pool = pd.DataFrame(
        [(date, instrument) for date in pool_dates for instrument in INSTRUMENTS],
        columns=["date", "instrument"],
    )
    exposures = pool.copy()
    exposures["float_market_cap"] = 1_000.0
    return pd.DataFrame(rows), pool, exposures


class OapBatch1PvTest(unittest.TestCase):
    def test_registered_oap_directions(self) -> None:
        bars, _ = _pv_inputs()
        max_ret = compute_pv_003_daily(bars, window=21, min_periods=10)
        max_last = max_ret.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertLess(max_last.loc["A", "factor_raw"], max_last.loc["C", "factor_raw"])

        skew = compute_pv_004_daily(bars, window=21, min_periods=10)
        skew_last = skew.groupby("instrument", sort=False).tail(1).set_index("instrument")
        self.assertLess(skew_last.loc["A", "factor_raw"], skew_last.loc["B", "factor_raw"])

        illiquidity = compute_pv_005_daily(bars, window=21, min_periods=10)
        illiq_last = illiquidity.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(
            illiq_last.loc["A", "factor_raw"],
            illiq_last.loc["B", "factor_raw"],
        )

        spread = compute_pv_006_daily(bars, window=21, min_periods=10)
        spread_last = spread.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(
            spread_last.loc["A", "factor_raw"],
            spread_last.loc["C", "factor_raw"],
        )

        zero_trade = compute_pv_007_daily(bars, window=21, min_periods=10)
        zero_last = zero_trade.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(
            zero_last.loc["A", "factor_raw"],
            zero_last.loc["B", "factor_raw"],
        )

    def test_all_pv_outputs_and_future_leakage(self) -> None:
        bars, pool = _pv_inputs()
        builders = (
            build_pv_003_factor,
            build_pv_004_factor,
            build_pv_005_factor,
            build_pv_006_factor,
            build_pv_007_factor,
        )
        for builder in builders:
            with self.subTest(builder=builder.__name__):
                result = builder(bars, pool, window=10, min_periods=5)
                self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
                self.assertEqual(len(result), len(pool))
                self.assertTrue(np.isfinite(result["factor"]).all())

        cutoff = pd.Timestamp("2024-01-26")
        original = build_pv_003_factor(bars, pool, window=10, min_periods=5)
        changed = bars.copy()
        changed.loc[changed["date"] > cutoff, "close"] *= 5.0
        rerun = build_pv_003_factor(changed, pool, window=10, min_periods=5)
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


class OapBatch1FrTest(unittest.TestCase):
    def test_asset_and_revenue_directions(self) -> None:
        financial, _, _ = _financial_inputs()
        asset = compute_fr_003_events(financial).dropna(subset=["factor_raw"])
        asset_latest = asset.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertLess(
            asset_latest.loc["A", "factor_raw"],
            asset_latest.loc["C", "factor_raw"],
        )

        revenue = compute_fr_004_events(financial).dropna(subset=["factor_raw"])
        revenue_latest = revenue.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(
            revenue_latest.loc["A", "factor_raw"],
            revenue_latest.loc["C", "factor_raw"],
        )

    def test_all_fr_outputs_and_future_leakage(self) -> None:
        financial, pool, exposures = _financial_inputs()
        results = (
            build_fr_003_factor(financial, pool),
            build_fr_004_factor(financial, pool),
            build_fr_005_factor(financial, exposures, pool),
        )
        for result in results:
            self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
            self.assertEqual(len(result), len(pool))
            self.assertTrue(np.isfinite(result["factor"]).all())

        asset = results[0]
        changed = financial.copy()
        future_rows = []
        for instrument in INSTRUMENTS:
            future_rows.append(
                {
                    "disclosure_date": pd.Timestamp("2024-04-10"),
                    "effective_date": pd.Timestamp("2024-04-11"),
                    "instrument": instrument,
                    "report_date": pd.Timestamp("2024-03-31"),
                    "category": "lf",
                    "shift": 0,
                    "net_cffoa": np.nan,
                    "net_profit": np.nan,
                    "operating_revenue": np.nan,
                    "total_assets": 10_000.0,
                }
            )
        changed = pd.concat([changed, pd.DataFrame(future_rows)], ignore_index=True)
        rerun = build_fr_003_factor(changed, pool)
        pd.testing.assert_frame_equal(asset, rerun)

    def test_cfp_direction(self) -> None:
        financial, pool, exposures = _financial_inputs()
        result = build_fr_005_factor(financial, exposures, pool)
        latest = result.loc[result["date"].eq(result["date"].max())].set_index(
            "instrument"
        )
        self.assertGreater(latest.loc["A", "factor"], latest.loc["C", "factor"])


if __name__ == "__main__":
    unittest.main()
