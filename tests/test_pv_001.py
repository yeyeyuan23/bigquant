from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.pv.pv_001 import build_pv_001_factor


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=8)
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    bar_rows: list[dict[str, object]] = []
    pool_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(dates):
        for stock_index, instrument in enumerate(instruments):
            pool_rows.append({"date": day, "instrument": instrument})
            if day_index == 6 and instrument == "000003.SZ":
                continue
            pre_close = 10.0 + stock_index
            direction = 1.0 if stock_index != 1 else -1.0
            for minute_index, minute in enumerate(("09:31", "15:00")):
                close = pre_close * (
                    1.0 + direction * (day_index + 1) * (minute_index + 1) / 1000.0
                )
                bar_rows.append(
                    {
                        "date": pd.Timestamp(f"{day.date()} {minute}"),
                        "instrument": instrument,
                        "high": close * 1.001,
                        "low": close * 0.999,
                        "close": close,
                        "pre_close": pre_close,
                        "amount": 100_000.0 * (day_index + 1) * (stock_index + 1),
                        "volume": 10_000.0 * (day_index + 1) * (stock_index + 1),
                        "deal_number": 100.0 * (day_index + 1) * (stock_index + 1),
                    }
                )
    return pd.DataFrame(bar_rows), pd.DataFrame(pool_rows)


class Pv001Test(unittest.TestCase):
    def test_output_contract_and_suspension_row(self) -> None:
        bars, pool = _synthetic_inputs()
        result = build_pv_001_factor(
            bars,
            pool,
            start_date="2024-01-05",
            end_date="2024-01-11",
            lookback=3,
            min_periods=2,
        )

        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(result["factor"]).all())
        expected_pool = pool.loc[
            pool["date"].between("2024-01-05", "2024-01-11")
        ]
        self.assertEqual(len(result), len(expected_pool))
        suspended = result.loc[
            result["date"].eq(pd.Timestamp("2024-01-10"))
            & result["instrument"].eq("000003.SZ")
        ]
        self.assertEqual(len(suspended), 1)

    def test_future_changes_do_not_change_past_factor(self) -> None:
        bars, pool = _synthetic_inputs()
        original = build_pv_001_factor(
            bars,
            pool,
            start_date="2024-01-05",
            end_date="2024-01-11",
            lookback=3,
            min_periods=2,
        )
        changed = bars.copy()
        changed.loc[changed["date"] > pd.Timestamp("2024-01-09"), "amount"] *= 100.0
        rerun = build_pv_001_factor(
            changed,
            pool,
            start_date="2024-01-05",
            end_date="2024-01-11",
            lookback=3,
            min_periods=2,
        )

        cutoff = pd.Timestamp("2024-01-09")
        pd.testing.assert_frame_equal(
            original.loc[original["date"] <= cutoff].reset_index(drop=True),
            rerun.loc[rerun["date"] <= cutoff].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
