from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.ob.ob_002 import build_ob_002_factor


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
            mid = 10.0 + stock_index
            for minute_index, minute in enumerate(times):
                row: dict[str, object] = {
                    "date": pd.Timestamp(f"{day.date()} {minute}"),
                    "instrument": instrument,
                }
                for level in range(1, 6):
                    row[f"bid_price{level}"] = mid - level * 0.01
                    row[f"ask_price{level}"] = mid + level * 0.01
                    row[f"bid_volume{level}"] = (
                        20_000 if level <= 2 and stock_index == 0 else 5_000
                    )
                    row[f"ask_volume{level}"] = (
                        20_000 if level <= 2 and stock_index == 1 else 5_000
                    )
                if minute_index % 50 == 0:
                    row["ask_price5"] = 0.0
                    row["ask_volume5"] = 0.0
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pool_rows)


class Ob002Test(unittest.TestCase):
    def test_output_and_future_leakage(self) -> None:
        bars, pool = _inputs()
        original = build_ob_002_factor(bars, pool)
        changed = bars.copy()
        cutoff = pd.Timestamp("2024-01-03")
        future = pd.to_datetime(changed["date"]).dt.normalize() > cutoff
        changed.loc[future, "bid_volume1"] *= 100.0
        rerun = build_ob_002_factor(changed, pool)
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
