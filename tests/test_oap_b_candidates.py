from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_008 import build_fr_008_factor
from bigalpha2026.candidates.fr.fr_009 import build_fr_009_factor
from bigalpha2026.candidates.fr.fr_010 import build_fr_010_factor
from bigalpha2026.candidates.fr.fr_011 import build_fr_011_factor
from bigalpha2026.candidates.pv.pv_013 import compute_pv_013_daily
from bigalpha2026.candidates.pv.pv_014 import compute_pv_014_daily
from bigalpha2026.candidates.pv.pv_015 import compute_pv_015_monthly
from bigalpha2026.candidates.pv.pv_016 import compute_pv_016_monthly
from bigalpha2026.candidates.pv.pv_017 import compute_pv_017_monthly
from bigalpha2026.candidates.pv.pv_018 import compute_pv_018_daily
from bigalpha2026.candidates.pv.pv_019 import compute_pv_019_daily


def daily_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2019-01-02", periods=90)
    rows: list[dict[str, object]] = []
    for stock_index, instrument in enumerate(("A", "B", "C")):
        price = 10.0
        for day_index, date in enumerate(dates):
            pre_close = price
            ret = 0.001 * (stock_index + 1) + 0.01 * np.sin(
                (day_index + stock_index) / 4
            )
            price *= 1.0 + ret
            rows.append(
                {
                    "date": date,
                    "instrument": instrument,
                    "close": price,
                    "pre_close": pre_close,
                    "volume": float(1000 + 10 * day_index + stock_index),
                    "turn": float(0.01 + 0.001 * stock_index + day_index / 10000),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, frame[["date", "instrument"]].copy()


def financial_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports = pd.date_range("2017-03-31", periods=28, freq="QE")
    rows: list[dict[str, object]] = []
    for stock_index, instrument in enumerate(("A", "B", "C")):
        for report_index, report_date in enumerate(reports):
            common = {
                "disclosure_date": report_date + pd.Timedelta(days=10),
                "effective_date": report_date + pd.Timedelta(days=11),
                "instrument": instrument,
                "report_date": report_date,
                "shift": 0,
            }
            rows.append(
                {
                    **common,
                    "category": "ttm",
                    "net_profit": 20 + stock_index + report_index,
                    "net_cffoa": 18 + stock_index + 0.8 * report_index,
                    "operating_revenue": 100 + 2 * stock_index + 3 * report_index,
                    "total_assets": np.nan,
                }
            )
            rows.append(
                {
                    **common,
                    "category": "lf",
                    "net_profit": np.nan,
                    "net_cffoa": np.nan,
                    "operating_revenue": np.nan,
                    "total_assets": 200 + 5 * stock_index + 4 * report_index,
                }
            )
    dates = pd.bdate_range("2023-01-03", periods=20)
    pool = pd.DataFrame(
        [(date, instrument) for date in dates for instrument in ("A", "B", "C")],
        columns=["date", "instrument"],
    )
    exposures = pool.copy()
    exposures["float_market_cap"] = exposures["instrument"].map(
        {"A": 500.0, "B": 600.0, "C": 700.0}
    )
    return pd.DataFrame(rows), pool, exposures


class OapBCandidatesTest(unittest.TestCase):
    def test_all_financial_outputs_follow_contract(self) -> None:
        financial, pool, exposures = financial_inputs()
        results = (
            build_fr_008_factor(financial, pool),
            build_fr_009_factor(financial, pool),
            build_fr_010_factor(financial, pool),
            build_fr_011_factor(financial, exposures, pool),
        )
        for result in results:
            self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
            self.assertEqual(len(result), len(pool))
            self.assertTrue(np.isfinite(result["factor"]).all())

    def test_all_price_components_preserve_long_windows(self) -> None:
        daily, _ = daily_inputs()
        outputs = (
            compute_pv_013_daily(daily, old_lag=20, recent_lag=5),
            compute_pv_014_daily(daily, window=10, min_periods=6),
            compute_pv_015_monthly(daily, window=3, min_periods=2),
            compute_pv_016_monthly(daily, window=3, min_periods=2),
            compute_pv_017_monthly(daily, window=3, min_periods=2),
            compute_pv_018_daily(daily, old_lag=30, recent_lag=10),
            compute_pv_019_daily(
                daily,
                beta_window=10,
                beta_min_periods=6,
                momentum_window=10,
                momentum_min_periods=6,
                skip_days=2,
            ),
        )
        for output in outputs:
            self.assertIn("factor_raw", output.columns)
            self.assertFalse(output["factor_raw"].dropna().empty)


if __name__ == "__main__":
    unittest.main()
