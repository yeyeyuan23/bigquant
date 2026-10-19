from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.fr.fr_002 import (
    build_fr_002_factor,
    compute_fr_002_events,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    trading_days = pd.bdate_range("2024-01-02", "2024-05-10")
    disclosures = pd.to_datetime(
        ["2024-01-05", "2024-02-10", "2024-03-08", "2024-04-05"]
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
                    "operating_revenue": 500.0 + event_index * 40 + stock_index,
                    "net_profit": 50.0 + event_index * 8 + stock_index,
                    "total_assets": np.nan,
                }
            )
            rows.append(
                {
                    **common,
                    "category": "lf",
                    "operating_revenue": np.nan,
                    "net_profit": np.nan,
                    "total_assets": 1_000.0 + stock_index * 100,
                }
            )
    pool_days = pd.bdate_range("2024-03-07", "2024-03-12")
    pool = pd.DataFrame(
        [(date, instrument) for date in pool_days for instrument in instruments],
        columns=["date", "instrument"],
    )
    pool["date"] = pool["date"].astype("datetime64[us]")
    return pd.DataFrame(rows), pool, trading_days


class Fr002Test(unittest.TestCase):
    def test_pit_and_output(self) -> None:
        financial, pool, trading_days = _inputs()
        events = compute_fr_002_events(financial, trading_days)
        weekend = events.loc[events["disclosure_date"].eq(pd.Timestamp("2024-02-10"))]
        self.assertTrue(weekend["effective_date"].eq(pd.Timestamp("2024-02-12")).all())
        result = build_fr_002_factor(financial, pool, trading_days)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(pool))
        self.assertTrue(np.isfinite(result["factor"]).all())

    def test_future_disclosure_does_not_change_past(self) -> None:
        financial, pool, trading_days = _inputs()
        original = build_fr_002_factor(financial, pool, trading_days)
        changed = financial.copy()
        future = pd.to_datetime(changed["date"]) > pd.Timestamp("2024-03-08")
        changed.loc[future, "operating_revenue"] *= 100.0
        rerun = build_fr_002_factor(changed, pool, trading_days)
        pd.testing.assert_frame_equal(original, rerun)


if __name__ == "__main__":
    unittest.main()
