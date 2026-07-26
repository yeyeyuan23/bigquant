import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    FROZEN_TEST_YEAR,
    PIPELINE_NAMES,
    VALIDATION_2022_YEAR,
    VALIDATION_2023_YEAR,
    contract_summary,
    parse_args,
    required_paths,
    synthetic_contract_summary,
)
from scripts.run_first_round import (
    candidate_pool_frame,
    classify_candidates,
    parse_args as parse_first_round_args,
)


class WorkflowScriptTest(unittest.TestCase):
    def test_dynamic_combination_periods_and_check_mode_are_frozen(self):
        self.assertEqual(DEVELOPMENT_YEARS, (2019, 2020, 2021))
        self.assertEqual(VALIDATION_2022_YEAR, 2022)
        self.assertEqual(VALIDATION_2023_YEAR, 2023)
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
        self.assertEqual(summary["factorlib_features"], 15)
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

    def test_first_round_can_resume_without_recomputing_metrics(self):
        args = parse_first_round_args(
            ["--resume-metrics", "--skip-correlations"]
        )
        self.assertTrue(args.resume_metrics)
        self.assertTrue(args.skip_correlations)

    def test_single_factor_gate_requires_both_validation_years(self):
        rows = []
        for period in ("development", "validation_2022", "validation_2023"):
            for variant in ("raw_full", "neutral_full", "raw_tradable"):
                rows.append(
                    {
                        "candidate_id": "PV-TEST",
                        "period": period,
                        "variant": variant,
                        "label": "ret_close_to_close",
                        "rank_ic_mean": 0.02,
                        "rank_ic_t_stat": 3.0,
                        "group_monotonicity": 0.8,
                    }
                )
        metrics = pd.DataFrame(rows)
        stability = pd.DataFrame(
            {
                "candidate_id": ["PV-TEST"] * 4,
                "period": ["development"] * 4,
                "frequency": ["year"] * 4,
                "positive": [True, True, True, True],
            }
        )
        passed = classify_candidates(metrics, stability)[0]
        self.assertTrue(passed["single_factor_cross_regime_passed"])

        metrics.loc[
            metrics["period"].eq("validation_2023")
            & metrics["variant"].eq("raw_full"),
            "rank_ic_mean",
        ] = -0.01
        failed = classify_candidates(metrics, stability)[0]
        self.assertFalse(failed["single_factor_cross_regime_passed"])
        self.assertTrue(failed["validation_2023_failures"])


if __name__ == "__main__":
    unittest.main()
