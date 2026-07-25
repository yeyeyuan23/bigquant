from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from bigalpha2026.hf_pressure import main as hf_main
from bigalpha2026.quality_interaction import main as interaction_main
from bigalpha2026.synthetic import make_synthetic_datasources


class FactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasources = make_synthetic_datasources(
            days=12,
            instruments=10,
            minutes_per_day=8,
        )
        universe = cls.datasources["bigalpha_2026_instruments"]
        cls.start = universe["date"].min()
        cls.end = universe["date"].max()

    def _assert_valid(self, frame: pd.DataFrame) -> None:
        self.assertEqual(list(frame.columns), ["date", "instrument", "factor"])
        self.assertFalse(frame.duplicated(["date", "instrument"]).any())
        self.assertTrue(np.isfinite(frame["factor"]).all())
        expected = len(self.datasources["bigalpha_2026_instruments"])
        self.assertEqual(len(frame), expected)
        self.assertGreaterEqual(float(frame["factor"].notna().mean()), 0.95)

    def test_both_factor_entrypoints_return_exact_schema(self) -> None:
        self._assert_valid(hf_main(self.datasources, self.start, self.end))
        self._assert_valid(interaction_main(self.datasources, self.start, self.end))

    def test_future_bar_changes_do_not_change_past_factor(self) -> None:
        dates = sorted(self.datasources["bigalpha_2026_instruments"]["date"].unique())
        cutoff = pd.Timestamp(dates[-2])
        baseline = hf_main(self.datasources, self.start, cutoff)
        changed = {name: frame.copy() for name, frame in self.datasources.items()}
        future_mask = changed["bigalpha_2026_stock_bar1m"]["date"].dt.normalize() > cutoff
        changed["bigalpha_2026_stock_bar1m"].loc[future_mask, "close"] *= 10.0
        rerun = hf_main(changed, self.start, cutoff)
        pd.testing.assert_frame_equal(baseline, rerun)

    def test_future_financial_changes_do_not_change_past_interaction(self) -> None:
        dates = sorted(self.datasources["bigalpha_2026_instruments"]["date"].unique())
        cutoff = pd.Timestamp(dates[-2])
        baseline = interaction_main(self.datasources, self.start, cutoff)
        changed = {name: frame.copy() for name, frame in self.datasources.items()}
        future_mask = changed["bigalpha_2026_financial"]["date"] > cutoff
        changed["bigalpha_2026_financial"].loc[future_mask, "net_profit"] *= 100.0
        rerun = interaction_main(changed, self.start, cutoff)
        pd.testing.assert_frame_equal(baseline, rerun)


if __name__ == "__main__":
    unittest.main()

