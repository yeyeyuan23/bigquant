import unittest

import numpy as np
import pandas as pd

from bigalpha2026.evaluation import (
    FactorLibraryValidationConfig,
    factorlib_regularized_incremental_batch_validation,
    factorlib_regularized_incremental_validation,
)
from bigalpha2026.factorlib import (
    FACTORLIB_COLUMNS,
    FACTORLIB_FEATURE_COLUMNS,
    validate_factorlib_columns,
    validate_factorlib_frame,
    validate_factorlib_subset_frame,
)
from bigalpha2026.research_policy import factorlib_incremental_gate


class FactorLibraryTest(unittest.TestCase):
    def test_frozen_subset_contract_is_strict(self):
        columns = ("amount", "turn")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04"]),
                "instrument": ["A"],
                "amount": [1.0],
                "turn": [0.1],
            }
        )
        validate_factorlib_subset_frame(frame, columns)
        with self.assertRaisesRegex(ValueError, "extra"):
            validate_factorlib_subset_frame(
                frame.assign(volume=2.0),
                columns,
            )

    def test_live_schema_has_36_features_and_strict_keys(self):
        self.assertEqual(FACTORLIB_COLUMNS[:2], ("date", "instrument"))
        self.assertEqual(len(FACTORLIB_COLUMNS), 38)
        self.assertEqual(len(FACTORLIB_FEATURE_COLUMNS), 36)
        validate_factorlib_columns(FACTORLIB_COLUMNS)

        row = {
            column: 1.0
            for column in FACTORLIB_COLUMNS
            if column not in {"date", "instrument"}
        }
        duplicated = pd.DataFrame(
            [
                {"date": "2022-01-04", "instrument": "A", **row},
                {"date": "2022-01-04", "instrument": "A", **row},
            ],
            columns=FACTORLIB_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_factorlib_frame(duplicated)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_factorlib_columns(FACTORLIB_COLUMNS[:-1])
        null_key = duplicated.iloc[:1].copy()
        null_key.loc[:, "instrument"] = None
        with self.assertRaisesRegex(ValueError, "null"):
            validate_factorlib_frame(null_key)

    def test_regularized_validation_detects_incremental_candidate(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2021-01-04", periods=100)
        rows = []
        labels = []
        for date in dates:
            for stock in range(30):
                base = rng.normal()
                candidate = rng.normal()
                noise = rng.normal(scale=0.15)
                rows.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        "base": base,
                        "candidate": candidate,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        "ret_close_to_close": 0.2 * base + candidate + noise,
                    }
                )

        summary, weights, predictions = factorlib_regularized_incremental_validation(
            pd.DataFrame(rows),
            pd.DataFrame(labels),
            ["base"],
            ["candidate"],
            config=FactorLibraryValidationConfig(
                train_window_days=40,
                test_window_days=20,
                alpha=0.001,
                l1_ratio=0.5,
            ),
        )
        self.assertGreater(summary["oos_rank_ic_increment"], 0)
        self.assertEqual(summary["candidate_min_nonzero_window_ratio"], 1.0)
        self.assertFalse(weights.empty)
        self.assertFalse(predictions.empty)
        passed, reasons = factorlib_incremental_gate(summary)
        self.assertTrue(passed, reasons)

    def test_batch_validation_reuses_one_baseline_for_all_candidates(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2021-01-04", periods=80)
        rows = []
        labels = []
        for date in dates:
            for stock in range(20):
                base = rng.normal()
                candidate_a = rng.normal()
                candidate_b = rng.normal()
                rows.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        "base": base,
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        "ret_close_to_close": (
                            0.2 * base
                            + candidate_a
                            - 0.5 * candidate_b
                            + rng.normal(scale=0.1)
                        ),
                    }
                )

        summaries, baseline_weights, candidate_weights, predictions = (
            factorlib_regularized_incremental_batch_validation(
                pd.DataFrame(rows),
                pd.DataFrame(labels),
                ["base"],
                ["candidate_a", "candidate_b"],
                config=FactorLibraryValidationConfig(
                    train_window_days=40,
                    test_window_days=20,
                    alpha=0.001,
                    l1_ratio=0.5,
                ),
            )
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(baseline_weights), 2)
        self.assertEqual(len(candidate_weights), 4)
        baseline_by_candidate = predictions.pivot(
            index=["date", "instrument"],
            columns="candidate",
            values="baseline_prediction",
        )
        self.assertTrue(
            np.allclose(
                baseline_by_candidate["candidate_a"],
                baseline_by_candidate["candidate_b"],
            )
        )
        self.assertEqual(summaries["baseline_oos_rank_ic"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
