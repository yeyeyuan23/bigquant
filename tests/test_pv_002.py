from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.pv_002 import build_pv_002_factor, compute_pv_002_daily


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=3)
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(dates):
        for stock_index, instrument in enumerate(instruments):
            pool_rows.append({"date": day, "instrument": instrument})
            if day_index == 1 and instrument == "000003.SZ":
                continue
            pre_close = 10.0 + stock_index
            gap = 0.02 if stock_index == 0 else -0.02
            open_price = pre_close * (1.0 + gap)
            close_price = open_price * (1.0 - np.sign(gap) * 0.01)
            for minute, close in (("09:31", open_price), ("15:00", close_price)):
                rows.append(
                    {
                        "date": pd.Timestamp(f"{day.date()} {minute}"),
                        "instrument": instrument,
                        "open": open_price,
                        "close": close,
                        "pre_close": pre_close,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pool_rows)


class Pv002Test(unittest.TestCase):
    def test_gap_direction(self) -> None:
        bars, _ = _inputs()
        daily = compute_pv_002_daily(bars)
        positive_gap = daily.loc[daily["instrument"].eq("000001.SZ"), "factor_raw"]
        negative_gap = daily.loc[daily["instrument"].eq("000002.SZ"), "factor_raw"]
        self.assertTrue((positive_gap < 0).all())
        self.assertTrue((negative_gap > 0).all())

    def test_zero_gap_is_neutral_not_missing(self) -> None:
        bars, _ = _inputs()
        zero_gap = bars["instrument"].eq("000003.SZ")
        bars.loc[zero_gap, "open"] = bars.loc[zero_gap, "pre_close"]
        bars.loc[zero_gap, "close"] = bars.loc[zero_gap, "pre_close"]
        daily = compute_pv_002_daily(bars)
        values = daily.loc[daily["instrument"].eq("000003.SZ"), "factor_raw"]
        self.assertTrue(values.eq(0.0).all())

    def test_output_and_future_leakage(self) -> None:
        bars, pool = _inputs()
        original = build_pv_002_factor(bars, pool)
        changed = bars.copy()
        cutoff = pd.Timestamp("2024-01-03")
        future = pd.to_datetime(changed["date"]).dt.normalize() > cutoff
        changed.loc[future, "close"] *= 10.0
        rerun = build_pv_002_factor(changed, pool)
        self.assertEqual(list(original.columns), ["date", "instrument", "factor"])
        self.assertEqual(len(original), len(pool))
        self.assertTrue(np.isfinite(original["factor"]).all())
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
