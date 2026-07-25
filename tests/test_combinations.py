import unittest

import pandas as pd

from bigalpha2026.combinations import fixed_rank_blend, positive_ic_weights
from bigalpha2026.research_policy import fixed_weight_rank_combination


class CombinationTest(unittest.TestCase):
    def setUp(self):
        keys = {
            "date": pd.to_datetime(["2022-01-04"] * 10 + ["2022-01-05"] * 10),
            "instrument": [str(value) for value in range(10)] * 2,
        }
        self.fr = pd.DataFrame({**keys, "factor": list(range(10)) * 2})
        self.hf = pd.DataFrame({**keys, "factor": list(reversed(range(10))) * 2})

    def test_fixed_weight_rank_combination_has_exact_contract(self):
        result = fixed_weight_rank_combination(
            {"FR-002": self.fr, "HF-001": self.hf},
            {"FR-002": 0.75, "HF-001": 0.25},
        )
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertEqual(len(result), 20)
        self.assertTrue(result["factor"].between(-1, 1).all())

    def test_public_blend_matches_policy_implementation(self):
        weights = {"FR-002": 0.75, "HF-001": 0.25}
        expected = fixed_weight_rank_combination(
            {"FR-002": self.fr, "HF-001": self.hf},
            weights,
        )
        actual = fixed_rank_blend(
            {"FR-002": self.fr, "HF-001": self.hf},
            weights,
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_positive_ic_weights_favor_predictive_member(self):
        labels = self.fr[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = list(range(10)) * 2
        weights = positive_ic_weights(
            {"FR-002": self.fr, "HF-001": self.hf},
            labels,
        )
        self.assertAlmostEqual(weights["FR-002"], 1.0)
        self.assertAlmostEqual(weights["HF-001"], 0.0)


if __name__ == "__main__":
    unittest.main()
