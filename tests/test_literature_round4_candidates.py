from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.fr.fr_015 import (
    build_fr_015_factor,
    compute_fr_015_events,
)
from candidates.ob.ob_005 import (
    build_ob_005_factor_from_daily,
    compute_ob_005_daily,
)
from candidates.pv.pv_023 import (
    build_pv_023_factor,
    compute_pv_023_daily,
)


def _overnight_daytime_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=40)
    patterns = {
        "A": ((0.01, -0.005), (-0.01, 0.005)),
        "B": (
            (0.01, -0.005),
            (0.01, 0.005),
            (-0.01, -0.005),
            (-0.01, 0.005),
        ),
        "C": ((0.01, 0.005), (-0.01, -0.005)),
    }
    rows: list[dict[str, object]] = []
    for instrument, pattern in patterns.items():
        for index, date in enumerate(dates):
            overnight, intraday = pattern[index % len(pattern)]
            pre_close = 10.0
            open_price = pre_close * (1.0 + overnight)
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "open": open_price,
                    "close": open_price * (1.0 + intraday),
                    "pre_close": pre_close,
                }
            )
    bars = pd.DataFrame(rows)
    return bars, bars[["date", "instrument"]].copy()


class LiteratureRound4Pv023Test(unittest.TestCase):
    def test_abnormal_cooccurrence_direction_and_interface(self) -> None:
        bars, pool = _overnight_daytime_inputs()
        latest = (
            compute_pv_023_daily(bars)
            .groupby("instrument", sort=False)
            .tail(1)
            .set_index("instrument")
        )
        self.assertGreater(
            latest.loc["A", "factor_raw"],
            latest.loc["B", "factor_raw"],
        )
        self.assertGreater(
            latest.loc["B", "factor_raw"],
            latest.loc["C", "factor_raw"],
        )
        result = build_pv_023_factor(bars, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())


def _financial_row(
    instrument: str,
    report_date: pd.Timestamp,
    *,
    net_profit: float,
    operating_revenue: float,
) -> dict[str, object]:
    disclosure_date = report_date + pd.Timedelta(days=90)
    return {
        "disclosure_date": disclosure_date,
        "effective_date": disclosure_date + pd.offsets.BDay(1),
        "instrument": instrument,
        "report_date": report_date,
        "category": "ttm",
        "shift": 0,
        "net_profit": net_profit,
        "operating_revenue": operating_revenue,
    }


class LiteratureRound4FinancialTest(unittest.TestCase):
    def test_margin_improvement_direction_interface_and_future_safety(self) -> None:
        rows: list[dict[str, object]] = []
        margins = {
            "A": (0.10, 0.20),
            "B": (0.20, 0.10),
            "C": (0.15, 0.15),
        }
        for instrument, (old_margin, new_margin) in margins.items():
            rows.extend(
                [
                    _financial_row(
                        instrument,
                        pd.Timestamp("2022-12-31"),
                        net_profit=100.0 * old_margin,
                        operating_revenue=100.0,
                    ),
                    _financial_row(
                        instrument,
                        pd.Timestamp("2023-12-31"),
                        net_profit=100.0 * new_margin,
                        operating_revenue=100.0,
                    ),
                ]
            )
        financial = pd.DataFrame(rows)
        events = compute_fr_015_events(financial)
        latest = events.groupby("instrument", sort=False).tail(1).set_index(
            "instrument"
        )
        self.assertGreater(latest.loc["A", "factor_raw"], 0.0)
        self.assertLess(latest.loc["B", "factor_raw"], 0.0)

        date = financial["effective_date"].max()
        pool = pd.DataFrame(
            {"date": [date] * 3, "instrument": ["A", "B", "C"]}
        )
        result = build_fr_015_factor(financial, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertTrue(np.isfinite(result["factor"]).all())

        future = financial.groupby("instrument", sort=False).tail(1).copy()
        future["report_date"] += pd.DateOffset(years=1)
        future["disclosure_date"] += pd.DateOffset(years=1)
        future["effective_date"] += pd.DateOffset(years=1)
        pd.testing.assert_frame_equal(
            result,
            build_fr_015_factor(
                pd.concat([financial, future], ignore_index=True),
                pool,
            ),
        )


class LiteratureRound4Ob005Test(unittest.TestCase):
    def test_microprice_pressure_direction_and_interface(self) -> None:
        date = pd.Timestamp("2024-04-01")
        daily = pd.DataFrame(
            {
                "date": [date] * 3,
                "instrument": ["A", "B", "C"],
                "tail_60_microprice_gap_median": [0.4, -0.4, 0.0],
                "tail_60_microprice_gap_sign_consistency": [1.0, -1.0, 0.0],
            }
        )
        components = compute_ob_005_daily(daily).set_index("instrument")
        self.assertGreater(components.loc["A", "factor_raw"], 0.0)
        self.assertLess(components.loc["B", "factor_raw"], 0.0)
        result = build_ob_005_factor_from_daily(
            daily,
            daily[["date", "instrument"]],
        )
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        ranked = result.set_index("instrument")
        self.assertGreater(ranked.loc["A", "factor"], ranked.loc["B", "factor"])


def _surprise_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_growth = (0.05, 0.10, 0.02, 0.08, 0.03, 0.12, 0.04, 0.07, 0.06)
    latest_growth = {"A": 0.40, "B": -0.20, "C": 0.06}
    rows: list[dict[str, object]] = []
    for instrument, latest_instrument_growth in latest_growth.items():
        earnings = 100.0
        revenue = 1_000.0
        growth_path = (*prior_growth, latest_instrument_growth)
        rows.append(
            _financial_row(
                instrument,
                pd.Timestamp("2014-12-31"),
                net_profit=earnings,
                operating_revenue=revenue,
            )
        )
        for year, growth in zip(range(2015, 2025), growth_path, strict=True):
            earnings *= 1.0 + growth
            revenue *= 1.0 + growth
            rows.append(
                _financial_row(
                    instrument,
                    pd.Timestamp(year=year, month=12, day=31),
                    net_profit=earnings,
                    operating_revenue=revenue,
                )
            )
    financial = pd.DataFrame(rows)
    date = financial["effective_date"].max()
    pool = pd.DataFrame({"date": [date] * 3, "instrument": ["A", "B", "C"]})
    micro = pool.assign(
        full_day_relative_spread_median=[0.01, 0.01, 0.01],
        tail_60_relative_spread_median=[0.03, 0.01, 0.015],
        full_day_total_depth_median=[100.0, 100.0, 100.0],
        tail_60_total_depth_median=[40.0, 100.0, 80.0],
    )
    return financial, micro, pool


if __name__ == "__main__":
    unittest.main()
