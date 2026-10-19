from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from candidates.ob.ob_001 import build_ob_001_factor


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
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
                mid *= 1.0 + (-0.002 if minute_index % 40 == 0 else 0.00001)
                row: dict[str, object] = {
                    "date": pd.Timestamp(f"{day.date()} {minute}"),
                    "instrument": instrument,
                }
                for level in range(1, 6):
                    row[f"bid_price{level}"] = mid - level * 0.01
                    row[f"ask_price{level}"] = mid + level * 0.01
                    row[f"bid_volume{level}"] = 10_000 + minute_index + level
                    row[f"ask_volume{level}"] = 9_000 + minute_index + level
                if instrument == "000002.SZ" and minute_index % 30 == 0:
                    row["ask_price5"] = 0.0
                    row["ask_volume5"] = 0.0
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pool_rows)


class Ob001Test(unittest.TestCase):
    def test_output_contract_and_absent_depth(self) -> None:
        bars, pool = _synthetic_inputs()
        result = build_ob_001_factor(bars, pool)
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(pool))
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(result["factor"]).all())

    def test_future_changes_do_not_change_past(self) -> None:
        bars, pool = _synthetic_inputs()
        original = build_ob_001_factor(bars, pool)
        cutoff = pd.Timestamp("2024-01-03")
        changed = bars.copy()
        future = pd.to_datetime(changed["date"]).dt.normalize() > cutoff
        changed.loc[future, "bid_volume1"] *= 100.0
        rerun = build_ob_001_factor(changed, pool)
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
