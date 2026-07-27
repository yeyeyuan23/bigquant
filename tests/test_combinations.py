import unittest
from multiprocessing import get_context

import pandas as pd

from bigalpha2026.combinations import (
    _prepare_joint_model_frame,
    fixed_rank_blend,
    lightgbm_model_config,
    paired_factor_rank_ic_increment,
    walk_forward_elastic_net,
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
)
from bigalpha2026.research_policy import fixed_weight_rank_combination


def _run_lightgbm_smoke() -> None:
    dates = pd.to_datetime(
        ["2019-01-02"] * 120
        + ["2020-01-02"] * 120
        + ["2021-01-04"] * 120
    )
    values = list(range(120)) * 3
    panel = pd.DataFrame(
        {
            "date": dates,
            "instrument": [str(value) for value in range(120)] * 3,
            "FR-002": values,
            "HF-001": list(reversed(values[:120])) * 3,
        }
    )
    labels = panel[["date", "instrument"]].copy()
    labels["ret_close_to_close"] = panel["FR-002"] / 1000
    result = walk_forward_lightgbm(
        panel,
        labels,
        feature_columns=("FR-002", "HF-001"),
        prediction_years=(2020, 2021),
        train_window_days=1,
        test_window_days=1,
    )
    assert set(result["date"].dt.year) == {2020, 2021}
    assert list(result.columns) == ["date", "instrument", "factor"]
    assert not result.duplicated(["date", "instrument"]).any()
    assert result["factor"].between(-1, 1).all()


class CombinationTest(unittest.TestCase):
    def setUp(self):
        keys = {
            "date": pd.to_datetime(["2022-01-04"] * 10 + ["2022-01-05"] * 10),
            "instrument": [str(value) for value in range(10)] * 2,
        }
        self.fr = pd.DataFrame({**keys, "factor": list(range(10)) * 2})
        self.hf = pd.DataFrame({**keys, "factor": list(reversed(range(10))) * 2})

    def test_fixed_weight_rank_combination_has_exact_contract(self):
        result = fixed_weight_rank_combination(
            {"FR-002": self.fr, "HF-001": self.hf},
            {"FR-002": 0.75, "HF-001": 0.25},
        )
        self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertEqual(len(result), 20)
        self.assertTrue(result["factor"].between(-1, 1).all())

    def test_public_blend_matches_policy_implementation(self):
        weights = {"FR-002": 0.75, "HF-001": 0.25}
        expected = fixed_weight_rank_combination(
            {"FR-002": self.fr, "HF-001": self.hf},
            weights,
        )
        actual = fixed_rank_blend(
            {"FR-002": self.fr, "HF-001": self.hf},
            weights,
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_lightgbm_predictions_are_strictly_walk_forward(self):
        process = get_context("spawn").Process(target=_run_lightgbm_smoke)
        process.start()
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail("lightgbm smoke test timed out")
        self.assertEqual(process.exitcode, 0)

    def test_elastic_net_predictions_are_strictly_walk_forward(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 3
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 3,
                "public": values,
                "candidate": list(reversed(values[:120])) * 3,
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = panel["public"] / 1000
        result = walk_forward_elastic_net(
            panel,
            labels,
            feature_columns=("public", "candidate"),
            prediction_years=(2020, 2021),
            train_window_days=1,
            test_window_days=1,
            alpha=1e-6,
        )
        self.assertEqual(set(result["date"].dt.year), {2020, 2021})
        self.assertFalse(result.duplicated(["date", "instrument"]).any())
        self.assertTrue(result["factor"].between(-1, 1).all())

    def test_elastic_net_is_invariant_to_label_units_and_records_weights(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 3
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 3,
                "public": values,
                "candidate": list(reversed(values[:120])) * 3,
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = panel["public"] / 10_000
        scaled = labels.copy()
        scaled["ret_close_to_close"] *= 100.0
        base_prediction, weights = walk_forward_elastic_net_with_weights(
            panel,
            labels,
            feature_columns=("public", "candidate"),
            prediction_years=(2020, 2021),
            train_window_days=1,
            test_window_days=1,
        )
        scaled_prediction = walk_forward_elastic_net(
            panel,
            scaled,
            feature_columns=("public", "candidate"),
            prediction_years=(2020, 2021),
            train_window_days=1,
            test_window_days=1,
        )
        pd.testing.assert_frame_equal(base_prediction, scaled_prediction)
        self.assertEqual(
            list(weights.columns),
            [
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "public",
                "candidate",
            ],
        )
        self.assertTrue(weights["public"].ne(0).all())

    def test_elastic_net_keeps_dates_with_a_neutral_constant_feature(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 3
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 3,
                "public": values,
                "sparse_candidate": [0.0] * len(dates),
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = panel["public"] / 10_000
        result = walk_forward_elastic_net(
            panel,
            labels,
            feature_columns=("public", "sparse_candidate"),
            prediction_years=(2020, 2021),
            train_window_days=1,
            test_window_days=1,
        )
        self.assertEqual(set(result["date"].dt.year), {2020, 2021})
        self.assertEqual(result["date"].nunique(), 2)
        self.assertEqual(len(result), 240)

    def test_learned_models_share_centered_rank_samples(self):
        dates = pd.to_datetime(
            ["2022-01-04"] * 10 + ["2022-01-05"] * 10
        )
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(10)] * 2,
                "dense": list(range(10)) * 2,
                "constant": [1.0] * 20,
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = list(range(10)) * 2
        prepared = _prepare_joint_model_frame(
            panel,
            labels,
            feature_columns=("dense", "constant"),
            label_column="ret_close_to_close",
        )
        self.assertEqual(len(prepared), 20)
        self.assertEqual(prepared["date"].nunique(), 2)
        self.assertTrue(prepared["constant"].eq(0.0).all())
        daily = prepared.groupby("date", sort=False)
        self.assertTrue(daily["dense"].mean().abs().lt(1e-12).all())
        self.assertTrue(
            daily["ret_close_to_close"].mean().abs().lt(1e-12).all()
        )
        self.assertTrue(
            prepared.groupby("date", sort=False)["dense"]
            .apply(lambda values: values.is_monotonic_increasing)
            .all()
        )

    def test_lightgbm_config_is_shallow_and_deterministic(self):
        config = lightgbm_model_config()
        self.assertEqual(config["random_state"], 20260726)
        self.assertEqual(config["training"], "rolling_60_train_20_test")
        self.assertEqual(
            config["feature_transform"],
            "daily_centered_percentile_rank",
        )
        self.assertEqual(
            config["target_transform"],
            "daily_centered_percentile_rank",
        )
        self.assertEqual(
            config["monotone_constraints"],
            "all_features_positive",
        )

    def test_paired_tree_increment_uses_daily_oos_rank_ic(self):
        dates = pd.bdate_range("2021-01-04", periods=40)
        rows = [
            {"date": date, "instrument": str(stock)}
            for date in dates
            for stock in range(20)
        ]
        labels = pd.DataFrame(rows)
        labels["ret_close_to_close"] = list(range(20)) * len(dates)
        baseline = labels[["date", "instrument"]].copy()
        baseline["factor"] = -labels["ret_close_to_close"]
        augmented = labels[["date", "instrument"]].copy()
        augmented["factor"] = labels["ret_close_to_close"]
        summary = paired_factor_rank_ic_increment(
            baseline,
            augmented,
            labels,
            test_window_days=20,
        )
        self.assertGreater(summary["oos_rank_ic_increment"], 0)
        self.assertEqual(summary["positive_window_ratio"], 1.0)
        self.assertEqual(summary["positive_years"], 1.0)
        self.assertEqual(summary["oos_days"], 40.0)
        self.assertEqual(summary["windows"], 2.0)


if __name__ == "__main__":
    unittest.main()
