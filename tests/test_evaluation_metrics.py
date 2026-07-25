import unittest

import pandas as pd

from bigalpha2026.evaluation import (
    quantile_group_returns,
    turnover_adjusted_long_short_returns,
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

    def test_one_way_cost_is_charged_on_entry_rebalance_and_exit(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04"] * 10 + ["2022-01-05"] * 10),
                "instrument": [str(value) for value in range(10)] * 2,
                "factor": list(range(10)) * 2,
                "ret_close_to_close": [value / 100 for value in range(10)] * 2,
            }
        )
        result = turnover_adjusted_long_short_returns(
            frame,
            quantiles=5,
            one_way_cost_bps=20,
        )
        self.assertEqual(len(result), 2)
        self.assertGreater(result["turnover"].iloc[0], 0)
        self.assertGreater(result["turnover"].iloc[-1], 0)
        self.assertTrue((result["net_return"] <= result["gross_return"]).all())


if __name__ == "__main__":
    unittest.main()
