import unittest

import pandas as pd

from bigalpha2026.research_policy import (
    FROZEN_FACTORLIB_SCREENED_FEATURES,
    COMBINATION_ADMISSION_GATE,
    FORMAL_EVALUATION_POLICY,
    HF_OB_ACTIVATED_OPTIONAL_MONTHS,
    HF_OB_MANDATORY_MONTHS,
    HF_OB_MAX_OPTIONAL_MONTHS,
    HF_OB_OPTIONAL_MONTH_POOL,
    candidate_ids,
    dual_factorlib_admission,
    fixed_weight_rank_combination,
    technical_gate,
)


class ResearchPolicyTest(unittest.TestCase):
    def test_screened_factorlib_membership_is_frozen_at_15_of_36(self):
        self.assertEqual(len(FROZEN_FACTORLIB_SCREENED_FEATURES), 15)
        self.assertEqual(len(set(FROZEN_FACTORLIB_SCREENED_FEATURES)), 15)
        self.assertTrue(
            all(
                feature.startswith("factorlib__")
                for feature in FROZEN_FACTORLIB_SCREENED_FEATURES
            )
        )

    def test_all_registered_candidates_are_available_to_first_round(self):
        self.assertEqual(len(candidate_ids()), 16)
        self.assertEqual(len(candidate_ids("first_round")), 8)
        self.assertEqual(
            candidate_ids("oap_batch1"),
            (
                "PV-003",
                "PV-004",
                "PV-005",
                "PV-006",
                "PV-007",
                "FR-003",
                "FR-004",
                "FR-005",
            ),
        )

    def test_formal_periods_and_representative_months_are_frozen(self):
        self.assertEqual(FORMAL_EVALUATION_POLICY.development_start, "2019-01-01")
        self.assertEqual(FORMAL_EVALUATION_POLICY.development_end, "2021-12-31")
        self.assertEqual(FORMAL_EVALUATION_POLICY.selection_start, "2022-01-01")
        self.assertEqual(FORMAL_EVALUATION_POLICY.selection_end, "2022-12-31")
        self.assertEqual(FORMAL_EVALUATION_POLICY.confirmation_start, "2023-01-01")
        self.assertEqual(FORMAL_EVALUATION_POLICY.confirmation_end, "2023-12-31")
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
                "2024-02",
                "2024-08",
            ),
        )
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
                "2024-05",
                "2024-11",
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

    def test_dual_factorlib_admission_requires_screened_baseline(self):
        self.assertEqual(
            dual_factorlib_admission(
                technical_passed=True,
                all36_passed=True,
                screened_passed=True,
            ),
            {"classification": "core_candidate", "enters_training": True},
        )
        self.assertEqual(
            dual_factorlib_admission(
                technical_passed=True,
                all36_passed=False,
                screened_passed=True,
            ),
            {
                "classification": "overlap_aware_candidate",
                "enters_training": True,
            },
        )
        self.assertEqual(
            dual_factorlib_admission(
                technical_passed=True,
                all36_passed=True,
                screened_passed=False,
            ),
            {"classification": "rejected", "enters_training": False},
        )
        self.assertEqual(
            dual_factorlib_admission(
                technical_passed=False,
                all36_passed=True,
                screened_passed=True,
            ),
            {"classification": "technical_reject", "enters_training": False},
        )


if __name__ == "__main__":
    unittest.main()
