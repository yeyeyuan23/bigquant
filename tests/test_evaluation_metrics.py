import unittest

import pandas as pd

from bigalpha2026.evaluation import (
    long_short_returns,
    quantile_group_returns,
    rank_ic_series,
)


class EvaluationMetricTest(unittest.TestCase):
    def test_quantile_groups_are_ordered_low_to_high(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04"] * 10),
                "factor": range(10),
                "ret_close_to_close": [value / 100 for value in range(10)],
            }
        )
        groups = quantile_group_returns(frame, quantiles=5)
        self.assertEqual(list(groups.columns), [f"group_{i}" for i in range(1, 6)])
        self.assertTrue(groups.iloc[0].is_monotonic_increasing)

    def test_empty_metric_inputs_return_typed_empty_outputs(self):
        empty = pd.DataFrame(
            columns=["date", "factor", "ret_close_to_close"]
        )
        self.assertIsInstance(rank_ic_series(empty), pd.Series)
        self.assertIsInstance(long_short_returns(empty), pd.Series)
        groups = quantile_group_returns(empty)
        self.assertEqual(list(groups.columns), [f"group_{i}" for i in range(1, 6)])

if __name__ == "__main__":
    unittest.main()
