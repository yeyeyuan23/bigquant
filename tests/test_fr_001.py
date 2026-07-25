from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr_001 import build_fr_001_factor, compute_fr_001_events


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    trading_days = pd.bdate_range("2024-01-02", "2024-04-12")
    disclosures = (
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-02-10"),
        pd.Timestamp("2024-03-08"),
        pd.Timestamp("2024-04-05"),
    )
    rows: list[dict[str, object]] = []
    for stock_index, instrument in enumerate(instruments):
        for event_index, disclosure in enumerate(disclosures):
            common = {
                "date": disclosure,
                "instrument": instrument,
                "shift": 0,
                "report_date": disclosure - pd.Timedelta(days=30),
            }
            rows.append(
                {
                    **common,
                    "category": "ttm",
                    "net_cffoa": 100.0 + event_index * 20 + stock_index,
                    "net_profit": 80.0 + event_index * 5 + stock_index,
                    "total_assets": np.nan,
                }
            )
            rows.append(
                {
                    **common,
                    "category": "lf",
                    "net_cffoa": np.nan,
                    "net_profit": np.nan,
                    "total_assets": 1_000.0 + stock_index * 100,
                }
            )
    pool_days = pd.bdate_range("2024-03-07", "2024-03-12")
    pool = pd.DataFrame(
        [(date, instrument) for date in pool_days for instrument in instruments],
        columns=["date", "instrument"],
    )
    return pd.DataFrame(rows), pool, trading_days


class Fr001Test(unittest.TestCase):
    def test_weekend_disclosure_is_effective_next_trading_day(self) -> None:
        financial, _, trading_days = _synthetic_inputs()
        events = compute_fr_001_events(financial, trading_days)
        weekend = events.loc[events["disclosure_date"].eq(pd.Timestamp("2024-02-10"))]
        self.assertTrue(weekend["effective_date"].eq(pd.Timestamp("2024-02-12")).all())

    def test_output_contract(self) -> None:
        financial, pool, trading_days = _synthetic_inputs()
        result = build_fr_001_factor(financial, pool, trading_days)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(pool))
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(result["factor"]).all())

    def test_future_disclosure_does_not_change_past(self) -> None:
        financial, pool, trading_days = _synthetic_inputs()
        original = build_fr_001_factor(financial, pool, trading_days)
        changed = financial.copy()
        future = pd.to_datetime(changed["date"]) > pd.Timestamp("2024-03-08")
        changed.loc[future, "net_cffoa"] *= 100.0
        rerun = build_fr_001_factor(changed, pool, trading_days)
        pd.testing.assert_frame_equal(original, rerun)


if __name__ == "__main__":
    unittest.main()
