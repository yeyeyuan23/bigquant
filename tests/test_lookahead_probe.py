import unittest

import pandas as pd

from scripts.aistudio_submission_lookahead_probe import compare_prefixes


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


if __name__ == "__main__":
    unittest.main()
