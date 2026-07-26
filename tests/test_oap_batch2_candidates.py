from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_006 import (
    build_fr_006_factor,
    compute_fr_006_events,
)
from bigalpha2026.candidates.fr.fr_007 import (
    build_fr_007_factor,
    compute_fr_007_events,
)
from bigalpha2026.candidates.pv.pv_008 import build_pv_008_factor
from bigalpha2026.candidates.pv.pv_009 import build_pv_009_factor
from bigalpha2026.candidates.pv.pv_010 import build_pv_010_factor
from bigalpha2026.candidates.pv.pv_011 import build_pv_011_factor
from bigalpha2026.candidates.pv.pv_012 import build_pv_012_factor


def daily_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=30)
    instruments = ("A", "B", "C")
    rows: list[dict[str, object]] = []
    for stock_index, instrument in enumerate(instruments):
        price = 10.0
        for day_index, date in enumerate(dates):
            pre_close = price
            daily_return = (
                0.002 * (stock_index + 1)
                + 0.01 * np.sin((day_index + stock_index) / 3)
            )
            price = pre_close * (1.0 + daily_return)
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "close": price,
                    "pre_close": pre_close,
                }
            )
    bars = pd.DataFrame(rows)
    pool = bars[["date", "instrument"]].copy()
    exposures = pool.copy()
    exposures["industry_level1_code"] = exposures["instrument"]
    exposures["float_market_cap"] = exposures["instrument"].map(
        {"A": 100.0, "B": 200.0, "C": 300.0}
    )
    return bars, pool, exposures


def financial_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    instruments = ("A", "B", "C")
    reports = pd.date_range("2019-03-31", periods=6, freq="QE")
    rows: list[dict[str, object]] = []
    for stock_index, instrument in enumerate(instruments):
        for report_index, report_date in enumerate(reports):
            value = 10.0 + stock_index + (stock_index - 1) * report_index
            rows.append(
                {
                    "disclosure_date": report_date + pd.Timedelta(days=10),
                    "effective_date": report_date + pd.Timedelta(days=11),
                    "instrument": instrument,
                    "report_date": report_date,
                    "category": "ttm",
                    "shift": 0,
                    "net_profit": value,
                }
            )
    financial = pd.DataFrame(rows)
    dates = pd.bdate_range("2020-04-13", periods=5)
    pool = pd.DataFrame(
        [(date, instrument) for date in dates for instrument in instruments],
        columns=["date", "instrument"],
    )
    return financial, pool


class OapBatch2Test(unittest.TestCase):
    def test_financial_candidates_are_pit_and_finite(self) -> None:
        financial, pool = financial_inputs()
        self.assertFalse(
            compute_fr_006_events(financial)["factor_raw"].dropna().empty
        )
        self.assertFalse(
            compute_fr_007_events(financial)["factor_raw"].dropna().empty
        )
        for builder in (build_fr_006_factor, build_fr_007_factor):
            result = builder(financial, pool)
            self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
            self.assertTrue(np.isfinite(result["factor"]).all())

    def test_daily_candidates_accept_a_natural_warmup(self) -> None:
        bars, pool, exposures = daily_inputs()
        builders = (
            lambda: build_pv_008_factor(bars, pool, window=10, min_periods=6),
            lambda: build_pv_009_factor(
                bars,
                pool,
                window=10,
                min_periods=6,
                lags=2,
            ),
            lambda: build_pv_010_factor(bars, pool, window=10, min_periods=6),
            lambda: build_pv_011_factor(
                bars,
                pool,
                old_lag=10,
                recent_lag=5,
            ),
            lambda: build_pv_012_factor(
                bars,
                exposures,
                pool,
                window=10,
                min_periods=6,
            ),
        )
        for builder in builders:
            with self.subTest(builder=builder):
                result = builder()
                self.assertEqual(len(result), len(pool))
                self.assertTrue(np.isfinite(result["factor"]).all())


if __name__ == "__main__":
    unittest.main()
