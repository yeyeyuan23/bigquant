from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation import build_return_labels


class CalendarLabelTest(unittest.TestCase):
    def test_suspension_does_not_skip_to_later_trade(self) -> None:
        trading_days = pd.to_datetime(["2024-04-29", "2024-04-30", "2024-05-06"])
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-29", "2024-05-06"]),
                "instrument": ["000851.SZ", "000851.SZ"],
                "open": [10.0, 11.0],
                "close": [10.2, 11.2],
                "pre_close": [9.9, 10.2],
            }
        )
        labels = build_return_labels(prices, trading_days)
        first = labels.loc[labels["date"].eq(pd.Timestamp("2024-04-29"))].iloc[0]
        self.assertTrue(np.isnan(first["ret_close_to_close"]))

    def test_exact_next_day_return_uses_next_pre_close(self) -> None:
        trading_days = pd.to_datetime(["2024-01-02", "2024-01-03"])
        prices = pd.DataFrame(
            {
                "date": trading_days,
                "instrument": ["000001.SZ", "000001.SZ"],
                "open": [10.0, 10.2],
                "close": [10.1, 10.5],
                "pre_close": [9.9, 10.1],
            }
        )
        labels = build_return_labels(prices, trading_days)
        value = labels.loc[
            labels["date"].eq(pd.Timestamp("2024-01-02")),
            "ret_close_to_close",
        ].iloc[0]
        self.assertAlmostEqual(value, 10.5 / 10.1 - 1.0)


if __name__ == "__main__":
    unittest.main()
