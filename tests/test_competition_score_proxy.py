import unittest

import numpy as np
import pandas as pd

from bigalpha2026.competition_score_proxy import (
    CompetitionScoreConfig,
    CompetitionScoreReference,
    inserted_percentile,
)


class CompetitionScoreProxyTest(unittest.TestCase):
    def test_inserted_percentile_matches_average_rank(self):
        reference = [1.0, 2.0, 2.0, 4.0]
        actual = inserted_percentile(2.0, reference)
        expected = pd.Series([*reference, 2.0]).rank(
            method="average",
            pct=True,
        ).iloc[-1]
        self.assertAlmostEqual(actual, expected)

    def test_route_score_uses_external_references_and_exact_weights(self):
        reference, labels, baseline, augmented = self._synthetic_frames()
        config = CompetitionScoreConfig(
            train_window_days=20,
            step_days=10,
            minimum_score_days=20,
            stability_window_days=40,
            stability_step_days=20,
        )
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=config,
        )

        score = scorer.score(augmented)
        self.assertAlmostEqual(
            score["score_proxy"],
            0.30 * score["a_proxy"] + 0.70 * score["b_proxy"],
        )
        self.assertAlmostEqual(
            score["a_proxy"],
            np.mean(
                [
                    score["a_rank_ic_mean_percentile"],
                    score["a_rank_ic_ir_percentile"],
                    score["a_long_short_sharpe_percentile"],
                    score["a_stress_ic_ir_percentile"],
                ]
            ),
        )
        self.assertAlmostEqual(
            score["b_model_score"],
            score["b_mean_abs_weight"]
            / (score["b_std_abs_weight"] + config.coefficient_epsilon),
        )
        self.assertEqual(score["reference_factor_count"], len(columns))
        self.assertFalse(scorer.protocol()["b_positive_coefficients"])
        self.assertEqual(
            scorer.protocol()["reference"],
            "competition_reference_pool_base_proxy",
        )
        self.assertEqual(
            scorer.protocol()["undisclosed_parameter_status"],
            "local_proxy_assumptions",
        )

        increment = scorer.paired_increment(baseline, augmented)
        self.assertGreater(increment["delta_score_proxy"], 0)
        self.assertAlmostEqual(
            increment["delta_score_proxy"],
            increment["augmented_score_proxy"]
            - increment["baseline_score_proxy"],
        )
        self.assertGreater(increment["oos_rank_ic_increment"], 0)
        self.assertEqual(increment["score_years"], 1.0)
        self.assertGreater(increment["score_windows"], 0)
        for component in (
            "rank_ic_mean",
            "rank_ic_ir",
            "long_short_sharpe",
            "stress_ic_ir",
        ):
            self.assertIn(f"baseline_a_{component}", increment)
            self.assertIn(f"augmented_a_{component}", increment)
            self.assertIn(f"delta_a_{component}", increment)
            self.assertIn(f"baseline_a_{component}_percentile", increment)
            self.assertIn(f"augmented_a_{component}_percentile", increment)
            self.assertIn(f"delta_a_{component}_percentile", increment)
        self.assertIn("baseline_b_mean_abs_weight", increment)
        self.assertIn("augmented_b_mean_abs_weight", increment)
        self.assertIn("baseline_b_std_abs_weight", increment)
        self.assertIn("augmented_b_std_abs_weight", increment)

    def test_best_direction_flips_an_inverted_route_once(self):
        reference, labels, _, augmented = self._synthetic_frames()
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=CompetitionScoreConfig(
                train_window_days=20,
                step_days=10,
                minimum_score_days=20,
                stability_window_days=40,
                stability_step_days=20,
            ),
        )
        inverted = augmented.copy()
        inverted["factor"] = -inverted["factor"]
        selected = scorer.score_best_direction(inverted)
        self.assertEqual(selected["selected_direction"], -1.0)
        self.assertGreater(
            selected["negative_score_proxy"],
            selected["positive_score_proxy"],
        )
        self.assertAlmostEqual(
            selected["positive_b_proxy"],
            selected["negative_b_proxy"],
            places=10,
        )

    def test_joint_crowding_scores_all_routes_in_one_fit(self):
        reference, labels, baseline, augmented = self._synthetic_frames()
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=CompetitionScoreConfig(
                train_window_days=20,
                step_days=10,
                minimum_score_days=20,
                stability_window_days=40,
                stability_step_days=20,
            ),
        )
        scores = scorer.score_joint_routes(
            {
                "baseline_route": baseline,
                "augmented_route": augmented,
            }
        )
        self.assertEqual(
            set(scores),
            {"baseline_route", "augmented_route"},
        )
        for score in scores.values():
            self.assertEqual(score["joint_route_count"], 2.0)
            self.assertAlmostEqual(
                score["score_proxy"],
                0.30 * score["a_proxy"] + 0.70 * score["b_proxy"],
            )

    def test_joint_crowding_uses_common_scorable_stock_days(self):
        reference, labels, baseline, augmented = self._synthetic_frames()
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=CompetitionScoreConfig(
                train_window_days=20,
                step_days=10,
                minimum_score_days=20,
                stability_window_days=40,
                stability_step_days=20,
            ),
        )
        baseline = baseline.iloc[1:].reset_index(drop=True)
        augmented = augmented.drop(index=[10]).reset_index(drop=True)
        expected_rows = len(
            baseline[["date", "instrument"]].merge(
                augmented[["date", "instrument"]],
                on=["date", "instrument"],
                how="inner",
                validate="one_to_one",
            )
        )
        scores = scorer.score_joint_routes(
            {
                "baseline_route": baseline,
                "augmented_route": augmented,
            }
        )
        self.assertEqual(
            scores["baseline_route"]["joint_common_rows"],
            float(expected_rows),
        )
        self.assertEqual(
            scores["augmented_route"]["joint_common_rows"],
            float(expected_rows),
        )

    def test_route_score_rejects_missing_scorable_reference_rows(self):
        reference, labels, _, augmented = self._synthetic_frames()
        config = CompetitionScoreConfig(
            train_window_days=20,
            step_days=10,
            minimum_score_days=20,
            stability_window_days=40,
            stability_step_days=20,
        )
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=config,
        )
        with self.assertRaisesRegex(
            ValueError,
            "missing values on scorable all36 stock-days",
        ):
            scorer.score(augmented.iloc[1:])

    def test_paired_increment_requires_identical_stock_day_keys(self):
        reference, labels, baseline, augmented = self._synthetic_frames()
        config = CompetitionScoreConfig(
            train_window_days=20,
            step_days=10,
            minimum_score_days=20,
            stability_window_days=40,
            stability_step_days=20,
        )
        columns = tuple(
            column
            for column in reference
            if column not in {"date", "instrument"}
        )
        scorer = CompetitionScoreReference(
            reference,
            labels,
            None,
            columns,
            config=config,
        )
        with self.assertRaisesRegex(
            ValueError,
            "identical stock-day keys",
        ):
            scorer.paired_increment(baseline, augmented.iloc[1:])

    @staticmethod
    def _synthetic_frames():
        rng = np.random.default_rng(20260727)
        dates = pd.bdate_range("2020-01-02", periods=80)
        reference_rows = []
        label_rows = []
        baseline_rows = []
        augmented_rows = []
        for date in dates:
            latent = rng.normal(size=30)
            for stock, signal in enumerate(latent):
                key = {
                    "date": date,
                    "instrument": f"S{stock:03d}",
                }
                reference_rows.append(
                    {
                        **key,
                        "factorlib__r1": rng.normal(),
                        "factorlib__r2": rng.normal(),
                        "factorlib__r3": rng.normal(),
                        "factorlib__r4": rng.normal(),
                    }
                )
                label_rows.append(
                    {
                        **key,
                        "ret_close_to_close": signal
                        + rng.normal(scale=0.15),
                    }
                )
                baseline_rows.append(
                    {
                        **key,
                        "factor": rng.normal(),
                    }
                )
                augmented_rows.append(
                    {
                        **key,
                        "factor": signal + rng.normal(scale=0.05),
                    }
                )
        return (
            pd.DataFrame(reference_rows),
            pd.DataFrame(label_rows),
            pd.DataFrame(baseline_rows),
            pd.DataFrame(augmented_rows),
        )


if __name__ == "__main__":
    unittest.main()
