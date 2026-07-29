import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from bigalpha2026.factor_pool import (
    validate_candidate_pool_manifest,
    write_candidate_pool_manifest,
)
from bigalpha2026.incremental_admission import (
    candidate_incremental_entry_diagnostics,
    promote_frozen_incremental_pool,
    validated_frozen_incremental_pool,
)
from bigalpha2026.research_policy import competition_score_increment_gate
from bigalpha2026.single_factor_admission import (
    candidate_s_trial_diagnostics,
    classify_candidates,
    run_single_factor_route_admission,
)
from bigalpha2026.tree_admission import (
    candidate_tree_entry_diagnostics,
    promote_frozen_tree_pool,
    unresolved_tree_candidates,
    validated_frozen_tree_pool,
)
from scripts import run_combinations
from scripts.run_combinations import (
    DEVELOPMENT_YEARS,
    FROZEN_TEST_YEAR,
    PIPELINE_NAMES,
    VALIDATION_2022_YEAR,
    VALIDATION_2023_YEAR,
    cleanup_obsolete_reports,
    contract_summary,
    enters_family_equal_rank,
    parse_args,
    rank_submission_routes,
    required_paths,
    synthetic_contract_summary,
)
from scripts.run_first_round import (
    CANDIDATE_POOL_VERSION,
    candidate_pool_frame,
)
from scripts.run_first_round import (
    parse_args as parse_first_round_args,
)


class WorkflowScriptTest(unittest.TestCase):
    def test_cleanup_obsolete_combination_reports_removes_only_retired_names(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            retired_names = (
                "incremental_direct_pool.csv",
                "incremental_backward_admission.csv",
                "tree_pool_promotion.csv",
            )
            for filename in retired_names:
                (reports_dir / filename).write_text("stale\n", encoding="utf-8")
            stale_root_report = reports_dir / "tree_factorwise_admission.csv"
            stale_root_report.write_text("stale root\n", encoding="utf-8")
            current_report = reports_dir / "routes" / "tree_factorwise_admission.csv"
            current_report.parent.mkdir()
            current_report.write_text("current\n", encoding="utf-8")

            cleanup_obsolete_reports(reports_dir)

            for filename in retired_names:
                self.assertFalse((reports_dir / filename).exists())
            self.assertFalse(stale_root_report.exists())
            self.assertEqual(
                current_report.read_text(encoding="utf-8"),
                "current\n",
            )

    def test_new_tree_contract_cannot_bootstrap_from_an_old_report(self):
        pending = unresolved_tree_candidates(
            ("self__A", "self__B"),
            refresh_features=set(),
            prior_candidates={"self__A", "self__B"},
            evaluated_fingerprints={},
            current_fingerprints={"self__A": "a", "self__B": "b"},
            has_compatible_evaluation_state=False,
        )
        self.assertEqual(pending, ("self__A", "self__B"))

    def test_incremental_pool_promotes_entry_passed_candidates(self):
        frozen = ("self__FR-002",)
        updated, promoted = promote_frozen_incremental_pool(
            frozen,
            ("self__PV-TEST",),
        )
        self.assertTrue(promoted)
        self.assertEqual(updated, ("self__FR-002", "self__PV-TEST"))

    def test_J_gate_does_not_fall_back_to_positive_rank_ic(self):
        passed, reasons = competition_score_increment_gate(
            {
                "delta_score_proxy": -0.001,
                "oos_rank_ic_increment": 0.02,
                "score_days": 200,
                "score_windows": 9,
                "positive_score_window_ratio": 0.70,
                "positive_score_years": 2,
            }
        )
        self.assertFalse(passed)
        self.assertIn("score proxy increment is not positive", reasons[0])

    def test_positive_J_is_not_vetoed_by_stability_diagnostics(self):
        passed, reasons = competition_score_increment_gate(
            {
                "delta_score_proxy": 0.001,
                "score_days": 200,
                "score_weight_windows": 9,
                "score_windows": 0,
                "positive_score_window_ratio": 0.0,
                "positive_score_years": 0,
            }
        )
        self.assertTrue(passed, reasons)
        self.assertEqual(reasons, [])

    def test_S_frozen_state_rejects_silent_factor_change(self):
        class FakeScoreReference:
            def protocol(self):
                return {"reference": "all36-test"}

            def score(self, _factor):
                return {
                    "a_proxy": 0.6,
                    "b_proxy": 0.6,
                    "score_proxy": 0.6,
                }

        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        panel = pd.DataFrame(
            {
                "date": dates.repeat(3),
                "instrument": ["A", "B", "C"] * 2,
                "self__PV-TEST": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "frozen_state.json"
            result = run_single_factor_route_admission(
                panel,
                FakeScoreReference(),
                ("self__PV-TEST",),
                frozen_state_path=state_path,
            )
            self.assertEqual(
                result.admitted_candidates,
                ("self__PV-TEST",),
            )
            changed = panel.copy()
            changed.loc[0, "self__PV-TEST"] = 9.0
            with self.assertRaisesRegex(
                RuntimeError,
                "frozen S state is incompatible",
            ):
                run_single_factor_route_admission(
                    changed,
                    FakeScoreReference(),
                    ("self__PV-TEST",),
                    frozen_state_path=state_path,
                )

    def test_S_bootstrap_uses_trial_gate_without_existing_baseline(self):
        class BootstrapScoreReference:
            def protocol(self):
                return {"reference": "all36-test"}

            def score(self, _factor):
                raise AssertionError("bootstrap S should not require absolute route J")

            def paired_increment(self, _baseline, _augmented, *, include_stability):
                raise AssertionError("bootstrap S has no baseline for paired J")

        dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
        panel = pd.DataFrame(
            {
                "date": dates.repeat(3),
                "instrument": ["A", "B", "C"] * 2,
                "self__PV-TEST": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

        result = run_single_factor_route_admission(
            panel,
            BootstrapScoreReference(),
            ("self__PV-TEST",),
            frozen_state_path=None,
        )

        self.assertEqual(result.admitted_candidates, ("self__PV-TEST",))
        self.assertTrue(result.evaluations[0]["s_route_passed"])
        self.assertNotIn("candidate_route_J_computed", result.evaluations[0])

    def test_frozen_incremental_pool_rejects_implicit_content_change(self):
        state = {
            "frozen_candidates": ["self__FR-002"],
            "candidate_fingerprints": {"self__FR-002": "old"},
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "cannot be changed implicitly",
        ):
            validated_frozen_incremental_pool(
                state,
                available_candidates=("self__FR-002",),
                candidate_fingerprints={"self__FR-002": "new"},
            )

    def test_tree_pool_changes_only_after_both_confirmation_gates(self):
        frozen = ("self__HF-002",)
        unchanged, promoted = promote_frozen_tree_pool(
            frozen,
            ("self__PV-TEST",),
            provisional_group_passed=True,
            relative_to_frozen_passed=False,
        )
        self.assertFalse(promoted)
        self.assertEqual(unchanged, frozen)

        updated, promoted = promote_frozen_tree_pool(
            frozen,
            ("self__PV-TEST",),
            provisional_group_passed=True,
            relative_to_frozen_passed=True,
        )
        self.assertTrue(promoted)
        self.assertEqual(updated, ("self__HF-002", "self__PV-TEST"))

    def test_tree_entry_allows_orthogonal_candidate_without_positive_ic(self):
        dates = pd.date_range("2021-01-01", periods=8, freq="D")
        rows = []
        labels = []
        for date in dates:
            rows.extend(
                [
                    {
                        "date": date,
                        "instrument": "A",
                        "factorlib__base": 1.0,
                        "self__orthogonal": 2.0,
                    },
                    {
                        "date": date,
                        "instrument": "B",
                        "factorlib__base": 2.0,
                        "self__orthogonal": 1.0,
                    },
                    {
                        "date": date,
                        "instrument": "C",
                        "factorlib__base": 3.0,
                        "self__orthogonal": 2.0,
                    },
                ]
            )
            labels.extend(
                [
                    {"date": date, "instrument": "A", "ret_close_to_close": 0.0},
                    {"date": date, "instrument": "B", "ret_close_to_close": 0.0},
                    {"date": date, "instrument": "C", "ret_close_to_close": 0.0},
                ]
            )
        diagnostics = candidate_tree_entry_diagnostics(
            pd.DataFrame(rows),
            pd.DataFrame(labels),
            candidate="self__orthogonal",
            baseline_columns=("factorlib__base",),
        )

        self.assertFalse(diagnostics["single_effect_passed"])
        self.assertTrue(diagnostics["orthogonal_passed"])
        self.assertTrue(diagnostics["tree_entry_passed"])

    def test_frozen_tree_pool_rejects_implicit_content_change(self):
        state = {
            "frozen_candidates": ["self__HF-002"],
            "candidate_fingerprints": {"self__HF-002": "old"},
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "cannot be changed implicitly",
        ):
            validated_frozen_tree_pool(
                state,
                eligible_candidates=("self__HF-002",),
                candidate_fingerprints={"self__HF-002": "new"},
            )
        self.assertEqual(
            validated_frozen_tree_pool(
                state,
                eligible_candidates=("self__HF-002",),
                candidate_fingerprints={"self__HF-002": "old"},
            ),
            ("self__HF-002",),
        )

    def test_composite_candidates_do_not_enter_family_equal_rank(self):
        self.assertTrue(enters_family_equal_rank("self__FR-013"))
        self.assertTrue(enters_family_equal_rank("self__PV-021"))
        self.assertFalse(enters_family_equal_rank("self__INT-002"))
        self.assertFalse(enters_family_equal_rank("factorlib__amount"))

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
        cache_args = parse_args(
            [
                "--incremental-cache-dir",
                "local-I-cache",
                "--refresh-incremental-cache",
                "--refresh-incremental-candidate",
                "PV-020",
                "--tree-cache-dir",
                "local-tree-cache",
                "--refresh-tree-cache",
                "--refresh-tree-candidate",
                "OB-003",
            ]
        )
        self.assertEqual(
            cache_args.incremental_cache_dir,
            Path("local-I-cache"),
        )
        self.assertTrue(cache_args.refresh_incremental_cache)
        self.assertEqual(
            cache_args.refresh_incremental_candidate,
            ["PV-020"],
        )
        self.assertEqual(cache_args.tree_cache_dir, Path("local-tree-cache"))
        self.assertTrue(cache_args.refresh_tree_cache)
        self.assertEqual(cache_args.refresh_tree_candidate, ["OB-003"])
        summary = synthetic_contract_summary()
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["factorlib_features"], 15)
        self.assertEqual(summary["factorlib_screened_features"], 15)
        self.assertEqual(
            summary["competition_J_reference"],
            "factorlib_all36_plus_j_baseline_candidates",
        )
        self.assertEqual(summary["competition_J_reference_features"], 37)
        self.assertEqual(summary["self_features"], 1)
        self.assertEqual(summary["j_baseline_features"], 1)
        self.assertEqual(
            {
                key
                for key in summary
                if key.endswith("_contract")
            },
            {f"{pipeline}_contract" for pipeline in PIPELINE_NAMES},
        )

    def test_j_baseline_columns_follow_candidate_metadata(self):
        columns = (
            "self__HF-001",
            "self__HF-039",
            "self__HF-043",
        )

        self.assertEqual(
            run_combinations.j_baseline_columns_from_self_columns(columns),
            ("self__HF-001", "self__HF-043"),
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

    def test_final_J_ranking_requires_all36_for_both_validation_years(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = required_paths(root / "data", root / "reports")
        all36_paths = {
            path
            for path in paths
            if "FACTORLIB_ALL36" in str(path) and path.suffix == ".parquet"
        }
        self.assertTrue(
            any("year=2022" in str(path) for path in all36_paths)
        )
        self.assertTrue(
            any("year=2023" in str(path) for path in all36_paths)
        )

    def test_final_route_order_uses_J_even_when_rank_ic_disagrees(self):
        decisions = [
            {
                "experiment": "self_factor_composite",
                "score_ranking_eligible": True,
                "robust_score_proxy": 0.42,
                "validation_combined_base_score_proxy": 0.45,
                "cross_regime_worst_year_rank_ic": 0.05,
            },
            {
                "experiment": "joint_elastic_net",
                "score_ranking_eligible": True,
                "robust_score_proxy": 0.58,
                "validation_combined_base_score_proxy": 0.60,
                "cross_regime_worst_year_rank_ic": -0.01,
            },
        ]
        self.assertEqual(
            rank_submission_routes(decisions),
            ["joint_elastic_net", "self_factor_composite"],
        )

    def test_validation_pipelines_allow_empty_s_pool(self):
        dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
        factor = pd.DataFrame(
            {
                "date": dates,
                "instrument": ["A", "A"],
                "factor": [0.1, 0.2],
            }
        )
        oriented = pd.DataFrame(
            {
                "date": dates,
                "instrument": ["A", "A"],
                "factorlib__base": [0.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": dates,
                "instrument": ["A", "A"],
                "ret_close_to_close": [0.0, 0.1],
            }
        )
        score_reference = SimpleNamespace(
            score_best_direction=lambda _factor: {
                "selected_direction": 1.0,
                "positive_score_proxy": 0.6,
                "negative_score_proxy": 0.4,
            },
            score=lambda _factor: {"score_proxy": 0.6},
            score_joint_routes=lambda routes: {
                route: {"score_proxy": 0.6} for route in routes
            },
        )
        incremental_result = SimpleNamespace(frozen_after=())
        tree_result = SimpleNamespace(
            admitted_candidates=(),
            predict_joint=lambda _years: factor.copy(),
        )

        with (
            patch.object(
                run_combinations,
                "walk_forward_elastic_net_with_weights",
                return_value=(factor.copy(), pd.DataFrame()),
            ),
            patch.object(
                run_combinations,
                "family_balanced_factor",
                side_effect=AssertionError("S should be skipped when empty"),
            ),
        ):
            pipelines, *_ = run_combinations.build_validation_pipelines(
                oriented,
                labels,
                ("factorlib__base",),
                score_reference,
                (),
                incremental_result,
                tree_result,
            )

        self.assertEqual(
            {experiment for experiment, _method in pipelines},
            {"joint_elastic_net", "joint_lightgbm"},
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
                        "long_short_mean": 0.001,
                    }
                )
        metrics = pd.DataFrame(rows)
        stability = pd.DataFrame(
            [
                *[
                    {
                        "candidate_id": "PV-TEST",
                        "period": "development",
                        "frequency": "month",
                        "positive": positive,
                    }
                    for positive in [True] * 6 + [False] * 4
                ],
                *[
                    {
                        "candidate_id": "PV-TEST",
                        "period": "development",
                        "frequency": "year",
                        "positive": positive,
                    }
                    for positive in (True, True, False)
                ],
            ]
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

    def test_s_trial_gate_requires_strong_stable_candidate(self):
        dates = pd.date_range("2021-01-01", periods=130, freq="D")
        rows = []
        labels = []
        for date in dates:
            for instrument, value in (
                ("A", 1.0),
                ("B", 2.0),
                ("C", 3.0),
                ("D", 4.0),
                ("E", 5.0),
            ):
                rows.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "self__strong": value,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "ret_close_to_close": value,
                    }
                )

        diagnostics = candidate_s_trial_diagnostics(
            pd.DataFrame(rows),
            pd.DataFrame(labels),
            candidate="self__strong",
            baseline_candidates=(),
        )

        self.assertTrue(diagnostics["s_quality_passed"])
        self.assertTrue(diagnostics["s_strength_passed"])
        self.assertTrue(diagnostics["s_stability_passed"])
        self.assertTrue(diagnostics["s_redundancy_passed"])
        self.assertTrue(diagnostics["s_trial_passed"])

    def test_i_entry_gate_accepts_residual_linear_signal(self):
        dates = pd.date_range("2021-01-01", periods=130, freq="D")
        rows = []
        labels = []
        values = {
            "A": (1.0, 0.0, -2.0),
            "B": (2.0, 5.0, 1.0),
            "C": (3.0, 8.0, 2.0),
            "D": (4.0, 9.0, 1.0),
            "E": (5.0, 8.0, -2.0),
        }
        for date in dates:
            for instrument, (base, candidate, label) in values.items():
                rows.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "factorlib__base": base,
                        "self__residual": candidate,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "instrument": instrument,
                        "ret_close_to_close": label,
                    }
                )
        diagnostics = candidate_incremental_entry_diagnostics(
            pd.DataFrame(rows),
            pd.DataFrame(labels),
            candidate="self__residual",
            baseline_columns=("factorlib__base",),
        )

        self.assertTrue(diagnostics["i_quality_passed"])
        self.assertTrue(diagnostics["i_residual_signal_passed"])
        self.assertTrue(diagnostics["i_trial_passed"])

    def test_single_factor_shape_is_diagnostic_not_a_gate(self):
        rows = []
        for period in ("development", "validation_2022", "validation_2023"):
            for variant in ("raw_full", "neutral_full", "raw_tradable"):
                rows.append(
                    {
                        "candidate_id": "HF-TEST",
                        "period": period,
                        "variant": variant,
                        "label": "ret_close_to_close",
                        "rank_ic_mean": 0.02,
                        "rank_ic_t_stat": 3.0,
                        "group_monotonicity": 0.2,
                        "long_short_mean": (
                            0.001 if variant == "neutral_full" else -0.001
                        ),
                    }
                )
        metrics = pd.DataFrame(rows)
        stability = pd.DataFrame(
            [
                *[
                    {
                        "candidate_id": "HF-TEST",
                        "period": "development",
                        "frequency": "month",
                        "positive": positive,
                    }
                    for positive in [True] * 6 + [False] * 4
                ],
                *[
                    {
                        "candidate_id": "HF-TEST",
                        "period": "development",
                        "frequency": "year",
                        "positive": positive,
                    }
                    for positive in (True, True, False)
                ],
            ]
        )
        decision = classify_candidates(metrics, stability)[0]
        self.assertTrue(decision["single_factor_cross_regime_passed"])
        self.assertTrue(
            decision["development_shape_evidence"][
                "neutral_long_short_return"
            ]
        )

        metrics.loc[
            metrics["period"].eq("development"),
            "long_short_mean",
        ] = -0.001
        decision = classify_candidates(metrics, stability)[0]
        self.assertTrue(decision["single_factor_cross_regime_passed"])
        self.assertFalse(
            decision["development_shape_evidence"][
                "neutral_long_short_return"
            ]
        )

    def test_single_factor_ic_and_stability_do_not_veto_route_J(self):
        rows = []
        for period in ("development", "validation_2022", "validation_2023"):
            for variant in ("raw_full", "neutral_full", "raw_tradable"):
                rows.append(
                    {
                        "candidate_id": "OB-INVERTED",
                        "period": period,
                        "variant": variant,
                        "label": "ret_close_to_close",
                        "rank_ic_mean": -0.02,
                        "rank_ic_t_stat": -3.0,
                        "group_monotonicity": 0.2,
                        "long_short_mean": -0.001,
                    }
                )
        stability = pd.DataFrame(
            [
                {
                    "candidate_id": "OB-INVERTED",
                    "period": "development",
                    "frequency": frequency,
                    "positive": False,
                }
                for frequency in ("month", "year")
            ]
        )
        decision = classify_candidates(
            pd.DataFrame(rows),
            stability,
        )[0]
        self.assertTrue(decision["single_factor_cross_regime_passed"])
        self.assertTrue(decision["development_failures"])
        self.assertTrue(
            decision["development_failures_are_diagnostic_only"]
        )


if __name__ == "__main__":
    unittest.main()
