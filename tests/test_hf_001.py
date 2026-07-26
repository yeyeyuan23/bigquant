from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.hf.hf_001 import (
    build_hf_001_factor,
    compute_hf_001_daily,
)


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=4)
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    times = pd.date_range("09:31", "11:30", freq="min").strftime("%H:%M").tolist()
    times += pd.date_range("13:01", "15:00", freq="min").strftime("%H:%M").tolist()
    rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(dates):
        for stock_index, instrument in enumerate(instruments):
            pool_rows.append({"date": day, "instrument": instrument})
            if day_index == 2 and instrument == "000003.SZ":
                continue
            price = 10.0 + stock_index
            for minute_index, minute in enumerate(times):
                shock = -0.015 if minute_index in {20, 50, 80, 150, 180, 210} else 0.0001
                recovery = 0.010 if minute_index in {25, 55, 85, 155, 185, 215} else 0.0
                price *= 1.0 + shock + recovery + day_index * 0.000001
                rows.append(
                    {
                        "date": pd.Timestamp(f"{day.date()} {minute}"),
                        "instrument": instrument,
                        "close": price,
                        "amount": 100_000.0 + minute_index * 100 + stock_index,
                        "volume": 10_000.0 + minute_index * 10 + stock_index,
                        "deal_number": 100.0 + minute_index + stock_index,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pool_rows)


class Hf001Test(unittest.TestCase):
    def test_output_contract_and_suspension(self) -> None:
        bars, pool = _synthetic_inputs()
        result = build_hf_001_factor(bars, pool)

        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(result), len(pool))
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(result["factor"]).all())

    def test_future_changes_do_not_change_past(self) -> None:
        bars, pool = _synthetic_inputs()
        original = build_hf_001_factor(bars, pool)
        changed = bars.copy()
        cutoff = pd.Timestamp("2024-01-04")
        future = pd.to_datetime(changed["date"]).dt.normalize() > cutoff
        changed.loc[future, "close"] *= 5.0
        rerun = build_hf_001_factor(changed, pool)
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )

    def test_lunch_break_is_not_a_minute_return(self) -> None:
        bars, _ = _synthetic_inputs()
        one_stock_day = bars.loc[
            bars["instrument"].eq("000001.SZ")
            & pd.to_datetime(bars["date"]).dt.normalize().eq(pd.Timestamp("2024-01-02"))
        ].copy()
        before = len(compute_hf_001_daily(one_stock_day, min_shocks=1))
        one_stock_day.loc[
            pd.to_datetime(one_stock_day["date"]).dt.strftime("%H:%M").eq("13:01"),
            "close",
        ] *= 100.0
        after = len(compute_hf_001_daily(one_stock_day, min_shocks=1))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
