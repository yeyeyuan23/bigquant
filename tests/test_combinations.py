import unittest
from multiprocessing import get_context
from unittest.mock import patch

import numpy as np
import pandas as pd

from bigalpha2026.combinations import (
    _causal_rolling_train_dates,
    _eligible_prediction_dates,
    _prepare_joint_model_frame,
    fixed_rank_blend,
    learned_model_training_config,
    lightgbm_model_config,
    paired_factor_rank_ic_increment,
    walk_forward_elastic_net,
    walk_forward_elastic_net_with_weights,
    walk_forward_lightgbm,
    walk_forward_lightgbm_with_importance,
)
from bigalpha2026.research_policy import fixed_weight_rank_combination


class _ZeroLightGBM:
    def __init__(self, fitted_targets: list[np.ndarray]):
        self.fitted_targets = fitted_targets
        self.booster_ = self
        self.feature_count = 0

    def fit(self, x, y):
        self.feature_count = x.shape[1]
        self.fitted_targets.append(np.asarray(y, dtype=float))
        return self

    def predict(self, x):
        return np.zeros(len(x), dtype=float)

    def feature_importance(self, importance_type="split"):
        return np.zeros(self.feature_count, dtype=float)


def _run_lightgbm_smoke() -> None:
    dates = pd.to_datetime(
        ["2019-01-02"] * 120
        + ["2019-01-03"] * 120
        + ["2020-01-02"] * 120
        + ["2021-01-04"] * 120
    )
    values = list(range(120)) * 4
    panel = pd.DataFrame(
        {
            "date": dates,
            "instrument": [str(value) for value in range(120)] * 4,
            "FR-002": values,
            "HF-001": list(reversed(values[:120])) * 4,
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


def _run_lightgbm_importance_smoke() -> None:
    dates = pd.to_datetime(
        ["2019-01-02"] * 120
        + ["2019-01-03"] * 120
        + ["2020-01-02"] * 120
        + ["2021-01-04"] * 120
    )
    values = list(range(120)) * 4
    panel = pd.DataFrame(
        {
            "date": dates,
            "instrument": [str(value) for value in range(120)] * 4,
            "FR-002": values,
            "HF-001": list(reversed(values[:120])) * 4,
        }
    )
    labels = panel[["date", "instrument"]].copy()
    labels["ret_close_to_close"] = panel["FR-002"] / 1000
    prediction, importance = walk_forward_lightgbm_with_importance(
        panel,
        labels,
        feature_columns=("FR-002", "HF-001"),
        prediction_years=(2020, 2021),
        train_window_days=1,
        test_window_days=1,
    )
    assert list(prediction.columns) == ["date", "instrument", "factor"]
    assert set(importance.columns) == {
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "feature",
        "split_importance",
        "gain_importance",
    }
    assert set(importance["feature"]) == {"FR-002", "HF-001"}
    assert importance["split_importance"].notna().all()
    assert importance["gain_importance"].notna().all()


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

    def test_lightgbm_importance_matches_walk_forward_windows(self):
        process = get_context("spawn").Process(
            target=_run_lightgbm_importance_smoke
        )
        process.start()
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail("lightgbm importance smoke test timed out")
        self.assertEqual(process.exitcode, 0)

    def test_elastic_net_predictions_are_strictly_walk_forward(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 120
            + ["2019-01-03"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 4
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 4,
                "public": values,
                "candidate": list(reversed(values[:120])) * 4,
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
            + ["2019-01-03"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 4
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 4,
                "public": values,
                "candidate": list(reversed(values[:120])) * 4,
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
            + ["2019-01-03"] * 120
            + ["2020-01-02"] * 120
            + ["2021-01-04"] * 120
        )
        values = list(range(120)) * 4
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(120)] * 4,
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
        self.assertEqual(
            config["training"],
            "causal_rolling_60_train_20_predict_label_embargo_1",
        )
        self.assertEqual(config["training_start_date"], "2019-01-01")
        self.assertEqual(config["train_window_days"], 60)
        self.assertEqual(config["prediction_block_days"], 20)
        self.assertEqual(config["label_embargo_days"], 1)
        self.assertEqual(
            config["feature_transform"],
            "daily_centered_percentile_rank",
        )
        self.assertEqual(
            config["target_transform"],
            "daily_centered_percentile_rank_residual_to_baseline",
        )
        self.assertEqual(
            config["monotone_constraints"],
            "all_self_features_positive",
        )
        self.assertEqual(
            config["residual_baseline_role"],
            "target_control_and_prediction_addback",
        )
        self.assertEqual(config["model_features"], "self_candidates_only")
        self.assertEqual(
            config["prediction_output"],
            "baseline_plus_residual_prediction",
        )

    def test_shared_training_contract_rolls_and_embargoes_last_label(self):
        all_dates = pd.date_range("2019-01-02", periods=8, freq="B")
        train_dates = _causal_rolling_train_dates(
            all_dates,
            7,
            train_window_days=4,
        )
        self.assertEqual(
            list(train_dates),
            list(all_dates[2:6]),
        )
        self.assertEqual(
            learned_model_training_config()["training"],
            "causal_rolling_60_train_20_predict_label_embargo_1",
        )

    def test_prediction_blocks_start_after_first_complete_train_window(self):
        all_dates = pd.date_range("2019-01-02", periods=85, freq="B")
        prediction_dates = _eligible_prediction_dates(
            all_dates,
            (2019,),
            train_window_days=60,
        )
        self.assertEqual(prediction_dates[0], all_dates[61])

    def test_lightgbm_screened_baseline_residualizes_and_is_added_back(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 10
            + ["2019-01-03"] * 10
            + ["2020-01-02"] * 10
        )
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(10)] * 3,
                "public": list(range(10)) * 3,
                "candidate": [0.0] * 30,
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = panel["public"]
        fitted_targets: list[np.ndarray] = []

        with patch(
            "bigalpha2026.combinations._lightgbm_regressor",
            return_value=_ZeroLightGBM(fitted_targets),
        ):
            result = walk_forward_lightgbm(
                panel,
                labels,
                feature_columns=("candidate",),
                prediction_years=(2020,),
                train_window_days=1,
                test_window_days=1,
                residual_baseline_columns=("public",),
            )

        self.assertEqual(len(fitted_targets), 1)
        self.assertTrue(np.allclose(fitted_targets[0], 0.0))
        expected = panel.loc[
            panel["date"].dt.year.eq(2020),
            ["date", "instrument"],
        ].copy()
        expected["date"] = expected["date"].astype("datetime64[ns]")
        expected["factor"] = np.tile(
            np.linspace(-0.8, 1.0, 10),
            len(expected) // 10,
        )
        expected = expected.sort_values(["date", "instrument"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_lightgbm_screened_baseline_addback_weight_changes_only_output(self):
        dates = pd.to_datetime(
            ["2019-01-02"] * 10
            + ["2019-01-03"] * 10
            + ["2020-01-02"] * 10
        )
        panel = pd.DataFrame(
            {
                "date": dates,
                "instrument": [str(value) for value in range(10)] * 3,
                "public": list(range(10)) * 3,
                "candidate": [0.0] * 30,
            }
        )
        labels = panel[["date", "instrument"]].copy()
        labels["ret_close_to_close"] = panel["public"]
        fitted_targets: list[np.ndarray] = []

        with patch(
            "bigalpha2026.combinations._lightgbm_regressor",
            return_value=_ZeroLightGBM(fitted_targets),
        ):
            result = walk_forward_lightgbm(
                panel,
                labels,
                feature_columns=("candidate",),
                prediction_years=(2020,),
                train_window_days=1,
                test_window_days=1,
                residual_baseline_columns=("public",),
                residual_baseline_addback_weight=0.0,
            )

        self.assertEqual(len(fitted_targets), 1)
        self.assertTrue(np.allclose(fitted_targets[0], 0.0))
        self.assertTrue(np.allclose(result["factor"], 0.1))

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
