from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.hf.hf_002 import build_hf_002_factor


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=3)
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    times = pd.date_range("09:31", "11:30", freq="min").strftime("%H:%M").tolist()
    times += pd.date_range("13:01", "15:00", freq="min").strftime("%H:%M").tolist()
    rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(dates):
        for stock_index, instrument in enumerate(instruments):
            pool_rows.append({"date": day, "instrument": instrument})
            if day_index == 1 and instrument == "000003.SZ":
                continue
            price = 10.0 + stock_index
            for minute_index, minute in enumerate(times):
                price *= 1.0 + (stock_index + 1) * 0.00001
                rows.append(
                    {
                        "date": pd.Timestamp(f"{day.date()} {minute}"),
                        "instrument": instrument,
                        "close": price,
                        "amount": 100_000.0 + minute_index,
                        "volume": 10_000.0 + minute_index,
                        "deal_number": 100.0 + stock_index * 20 + minute_index % 5,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pool_rows)


class Hf002Test(unittest.TestCase):
    def test_output_and_future_leakage(self) -> None:
        bars, pool = _inputs()
        original = build_hf_002_factor(bars, pool)
        changed = bars.copy()
        cutoff = pd.Timestamp("2024-01-03")
        future = pd.to_datetime(changed["date"]).dt.normalize() > cutoff
        changed.loc[future, "deal_number"] *= 100.0
        rerun = build_hf_002_factor(changed, pool)
        self.assertEqual(list(original.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(original), len(pool))
        self.assertFalse(original.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(original["factor"]).all())
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
