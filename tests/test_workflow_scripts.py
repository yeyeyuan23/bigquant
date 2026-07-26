import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bigalpha2026.factor_pool import (
    validate_candidate_pool_manifest,
    write_candidate_pool_manifest,
)
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
    CANDIDATE_POOL_VERSION,
    candidate_pool_frame,
    classify_candidates,
)
from scripts.run_first_round import (
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
            [
                "--resume-metrics",
                "--skip-correlations",
                "--refresh-candidate",
                "OB-001",
            ]
        )
        self.assertTrue(args.resume_metrics)
        self.assertTrue(args.skip_correlations)
        self.assertEqual(args.refresh_candidate, ["OB-001"])

    def test_candidate_pool_manifest_rejects_stale_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            parquet_path = data_root / "factors" / "candidate_pool.parquet"
            manifest_path = data_root / "manifest_candidate_pool.json"
            parquet_path.parent.mkdir(parents=True)
            pool = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
                    "instrument": ["A", "B"],
                    "candidate_id": ["PV-001", "PV-001"],
                    "factor_version": [CANDIDATE_POOL_VERSION] * 2,
                    "factor": [0.1, 0.2],
                }
            )
            pool.to_parquet(parquet_path, index=False)
            write_candidate_pool_manifest(
                pool,
                parquet_path=parquet_path,
                manifest_path=manifest_path,
                data_root=data_root,
            )
            validate_candidate_pool_manifest(
                pool,
                parquet_path=parquet_path,
                manifest_path=manifest_path,
                data_root=data_root,
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["factor_version"] = "stale-v1"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest version is stale"):
                validate_candidate_pool_manifest(
                    pool,
                    parquet_path=parquet_path,
                    manifest_path=manifest_path,
                    data_root=data_root,
                )

    def test_single_factor_gate_never_uses_validation_years_for_admission(self):
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
                "candidate_id": ["PV-TEST"] * 10,
                "period": ["development"] * 10,
                "frequency": ["month"] * 10,
                "positive": [True] * 6 + [False] * 4,
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
        self.assertTrue(failed["single_factor_cross_regime_passed"])
        self.assertTrue(failed["validation_2023_observations"])


if __name__ == "__main__":
    unittest.main()
