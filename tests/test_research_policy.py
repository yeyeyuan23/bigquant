import unittest

import pandas as pd

from bigalpha2026.research_policy import (
    COMBINATION_ADMISSION_GATE,
    FORMAL_EVALUATION_POLICY,
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    HF_OB_MAX_OPTIONAL_MONTHS,
    HF_OB_OPTIONAL_MONTH_POOL,
    HF_OB_REPRESENTATIVE_MONTHS,
    MINIMAL_BASELINE,
    candidate_ids,
    equal_weight_rank_combination,
    fixed_weight_rank_combination,
    incremental_gate,
    technical_gate,
)


class ResearchPolicyTest(unittest.TestCase):
    def test_minimal_baseline_is_frozen_locally(self):
        self.assertEqual(MINIMAL_BASELINE, ("PV-001", "HF-001"))
        self.assertEqual(
            candidate_ids("minimal_baseline"),
            ("PV-001", "HF-001"),
        )

    def test_formal_periods_and_representative_months_are_frozen(self):
        self.assertEqual(FORMAL_EVALUATION_POLICY.development_start, "2019-01-01")
        self.assertEqual(FORMAL_EVALUATION_POLICY.development_end, "2022-12-31")
        self.assertEqual(FORMAL_EVALUATION_POLICY.primary_label, "ret_close_to_close")
        self.assertEqual(
            HF_OB_MANDATORY_MONTHS,
            (
                "2019-02",
                "2019-08",
                "2020-02",
                "2020-08",
                "2021-02",
                "2021-08",
                "2022-02",
                "2022-08",
                "2023-02",
                "2023-08",
            ),
        )
        self.assertEqual(HF_OB_REPRESENTATIVE_MONTHS, HF_OB_MANDATORY_MONTHS)
        self.assertEqual(
            HF_OB_OPTIONAL_MONTH_POOL,
            (
                "2019-05",
                "2019-11",
                "2020-05",
                "2020-11",
                "2021-05",
                "2021-11",
                "2022-05",
                "2022-11",
                "2023-05",
                "2023-11",
            ),
        )
        self.assertEqual(HF_OB_MAX_OPTIONAL_MONTHS, 6)
        self.assertEqual(HF_OB_ACTIVATED_OPTIONAL_MONTHS, ("2022-11",))
        self.assertTrue(
            set(HF_OB_ACTIVATED_OPTIONAL_MONTHS).issubset(HF_OB_OPTIONAL_MONTH_POOL)
        )
        self.assertLessEqual(
            len(HF_OB_ACTIVATED_OPTIONAL_MONTHS), HF_OB_MAX_OPTIONAL_MONTHS
        )
        self.assertEqual(FORMAL_EVALUATION_POLICY.cost_sensitivity_bps, (0, 10, 20, 30))

    def test_equal_weight_rank_combination(self):
        keys = {
            "date": pd.to_datetime(["2022-01-04"] * 3),
            "instrument": ["A", "B", "C"],
        }
        pv = pd.DataFrame({**keys, "factor": [1.0, 2.0, 3.0]})
        hf = pd.DataFrame({**keys, "factor": [3.0, 2.0, 1.0]})
        result = equal_weight_rank_combination(
            {"PV-001": pv, "HF-001": hf}
        )
        self.assertEqual(result["factor"].nunique(), 1)
        self.assertAlmostEqual(float(result["factor"].iloc[0]), 1.0 / 3.0)

    def test_fixed_weight_combination_is_frozen_and_ranked(self):
        keys = {
            "date": pd.to_datetime(["2022-01-04"] * 3),
            "instrument": ["A", "B", "C"],
        }
        pv = pd.DataFrame({**keys, "factor": [1.0, 2.0, 3.0]})
        hf = pd.DataFrame({**keys, "factor": [3.0, 2.0, 1.0]})
        result = fixed_weight_rank_combination(
            {"PV-001": pv, "HF-001": hf},
            {"PV-001": 0.75, "HF-001": 0.25},
        )
        self.assertGreater(result.loc[result["instrument"].eq("C"), "factor"].iloc[0], 0)
        self.assertEqual(COMBINATION_ADMISSION_GATE.minimum_rank_ic_t_stat, 1.5)

    def test_gates_are_explicit(self):
        passed, reasons = technical_gate(0.99, 100, 60, 0.0)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        passed, reasons = incremental_gate(
            {
                "rank_ic_mean": 0.02,
                "rank_ic_ir": 0.2,
                "long_short_sharpe": 0.5,
            },
            {
                "rank_ic_mean": 0.019,
                "rank_ic_ir": 0.25,
                "long_short_sharpe": 0.45,
            },
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
