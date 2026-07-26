import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_combinations import (
    CONFIRMATION_YEAR,
    DEVELOPMENT_YEARS,
    FROZEN_TEST_YEAR,
    PIPELINE_NAMES,
    SELECTION_YEAR,
    contract_summary,
    parse_args,
    required_paths,
    synthetic_contract_summary,
)
from scripts.run_first_round import candidate_pool_frame


class WorkflowScriptTest(unittest.TestCase):
    def test_dynamic_combination_periods_and_check_mode_are_frozen(self):
        self.assertEqual(DEVELOPMENT_YEARS, (2019, 2020, 2021))
        self.assertEqual(SELECTION_YEAR, 2022)
        self.assertEqual(CONFIRMATION_YEAR, 2023)
        self.assertEqual(FROZEN_TEST_YEAR, 2024)
        self.assertEqual(
            PIPELINE_NAMES,
            (
                "self_factor_composite",
                "joint_elastic_net",
                "joint_lightgbm",
            ),
        )
        args = parse_args(["--check"])
        self.assertTrue(args.check)
        summary = synthetic_contract_summary()
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["factorlib_features"], 36)
        self.assertEqual(summary["factorlib_screened_features"], 15)
        self.assertEqual(summary["self_features"], 1)
        self.assertEqual(
            {
                key
                for key in summary
                if key.endswith("_contract")
            },
            {f"{pipeline}_contract" for pipeline in PIPELINE_NAMES},
        )

    def test_check_mode_reports_missing_inputs_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, loaded = contract_summary(root / "data", root / "reports")
        self.assertEqual(summary["status"], "missing_inputs")
        self.assertIsNone(loaded)
        self.assertEqual(
            len(summary["missing"]),
            len(required_paths(root / "data", root / "reports")),
        )

    def test_first_round_candidate_pool_has_stable_long_contract(self):
        factor = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
                "instrument": ["A", "B"],
                "factor": [0.1, 0.2],
            }
        )
        pool = candidate_pool_frame({"PV-001": factor}, factor_version="test-v1")
        self.assertEqual(
            list(pool.columns),
            [
                "date",
                "instrument",
                "candidate_id",
                "factor_version",
                "factor",
            ],
        )
        self.assertEqual(set(pool["candidate_id"]), {"PV-001"})
        self.assertEqual(set(pool["factor_version"]), {"test-v1"})


if __name__ == "__main__":
    unittest.main()
