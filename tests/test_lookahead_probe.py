import unittest

import numpy as np
import pandas as pd

from scripts.aistudio_submission_lookahead_probe import (
    compare_prefixes,
    normalized_factor,
    validate_output_contract,
)


class LookaheadProbeTest(unittest.TestCase):
    def test_identical_prefix_passes(self):
        full = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "instrument": ["A", "A"],
                "factor": [0.1, 0.2],
            }
        )
        cut = full.iloc[:1].copy()
        summary = compare_prefixes(
            full,
            cut,
            pd.Timestamp("2024-01-02"),
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["difference_rows"], 0)

    def test_changed_cutoff_value_is_reported(self):
        full = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "instrument": ["A"],
                "factor": [0.1],
            }
        )
        cut = full.assign(factor=0.9)
        summary = compare_prefixes(
            full,
            cut,
            pd.Timestamp("2024-01-02"),
        )
        self.assertEqual(summary["status"], "lookahead_suspected")
        self.assertEqual(summary["difference_rows"], 1)
        self.assertEqual(summary["first_difference_date"], "2024-01-02")

    def test_output_contract_rejects_a_whole_missing_date(self):
        expected = pd.DataFrame(
            {
                "date": [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ],
                "instrument": ["A", "B", "A", "B"],
            }
        )
        output = expected.iloc[:2].assign(factor=[0.1, -0.1])

        summary = validate_output_contract(
            output,
            expected,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        )

        self.assertEqual(summary["status"], "invalid_output_contract")
        self.assertEqual(summary["missing_date_count"], 1)
        self.assertEqual(summary["missing_date_sample"], ["2024-01-03"])

    def test_output_contract_rejects_whole_date_with_nan_factors(self):
        expected = pd.DataFrame(
            {
                "date": [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ],
                "instrument": ["A", "B", "A", "B"],
            }
        )
        output = expected.assign(factor=[0.1, -0.1, np.nan, np.nan])

        summary = validate_output_contract(
            output,
            expected,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        )

        self.assertEqual(summary["status"], "invalid_output_contract")
        self.assertEqual(summary["missing_date_count"], 1)

    def test_output_contract_accepts_neutral_warmup_date(self):
        expected = pd.DataFrame(
            {
                "date": [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ],
                "instrument": ["A", "B", "A", "B"],
            }
        )
        output = expected.assign(factor=[0.1, -0.1, 0.0, 0.0])

        summary = validate_output_contract(
            output,
            expected,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["missing_date_count"], 0)

    def test_normalization_rejects_duplicate_keys(self):
        output = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-02"],
                "instrument": ["A", "A"],
                "factor": [0.1, 0.2],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            normalized_factor(output, pd.Timestamp("2024-01-02"))


if __name__ == "__main__":
    unittest.main()
