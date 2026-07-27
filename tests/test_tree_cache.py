import tempfile
import unittest
from pathlib import Path

import pandas as pd

from bigalpha2026.tree_cache import (
    TREE_CACHE_SCHEMA_VERSION,
    TreePredictionCache,
    feature_fingerprints,
    frame_column_fingerprint,
)


class TreePredictionCacheTest(unittest.TestCase):
    def test_official_J_contract_uses_v3_schema(self):
        self.assertEqual(
            TREE_CACHE_SCHEMA_VERSION,
            "tree-prediction-cache-v3-J",
        )

    def setUp(self):
        self.panel = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05"]
                ),
                "instrument": ["A", "B", "A", "B"],
                "base": [0.1, 0.2, 0.3, 0.4],
                "candidate_a": [1.0, 2.0, 3.0, 4.0],
                "candidate_b": [4.0, 3.0, 2.0, 1.0],
            }
        )
        self.labels = self.panel[["date", "instrument"]].assign(
            ret_close_to_close=[0.01, 0.02, -0.01, 0.03]
        )
        self.prediction = self.panel[["date", "instrument"]].assign(
            factor=[-0.5, 0.5, 0.25, -0.25]
        )

    def cache(self, directory: Path, columns):
        return TreePredictionCache(
            directory,
            feature_fingerprints=feature_fingerprints(self.panel, columns),
            label_fingerprint=frame_column_fingerprint(
                self.labels,
                "ret_close_to_close",
            ),
            model_config={"model": "test-tree", "depth": 2},
        )

    def test_exact_prediction_is_reused_across_cache_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            calls = []
            first = self.cache(path, ("base", "candidate_a"))
            actual, hit, key = first.get_or_compute(
                ("base", "candidate_a"),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                compute=lambda: calls.append("fit") or self.prediction,
            )
            self.assertFalse(hit)
            self.assertEqual(calls, ["fit"])
            pd.testing.assert_frame_equal(actual, self.prediction)

            second = self.cache(path, ("base", "candidate_a"))
            cached, hit, second_key = second.get_or_compute(
                ("base", "candidate_a"),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                compute=lambda: calls.append("unexpected") or self.prediction,
            )
            self.assertTrue(hit)
            self.assertEqual(second_key, key)
            self.assertEqual(calls, ["fit"])
            pd.testing.assert_frame_equal(cached, self.prediction)

    def test_changed_factor_value_invalidates_affected_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = self.cache(path, ("base", "candidate_a"))
            _, _, old_key = first.get_or_compute(
                ("base", "candidate_a"),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                compute=lambda: self.prediction,
            )
            changed = self.panel.copy()
            changed.loc[0, "candidate_a"] = 99.0
            second = TreePredictionCache(
                path,
                feature_fingerprints=feature_fingerprints(
                    changed,
                    ("base", "candidate_a"),
                ),
                label_fingerprint=frame_column_fingerprint(
                    self.labels,
                    "ret_close_to_close",
                ),
                model_config={"model": "test-tree", "depth": 2},
            )
            calls = []
            _, hit, new_key = second.get_or_compute(
                ("base", "candidate_a"),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                compute=lambda: calls.append("refit") or self.prediction,
            )
            self.assertFalse(hit)
            self.assertNotEqual(new_key, old_key)
            self.assertEqual(calls, ["refit"])

    def test_new_candidate_changes_pool_state_but_not_public_baseline_key(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(
                Path(directory),
                ("base", "candidate_a", "candidate_b"),
            )
            old_state = cache.pool_state_digest(("candidate_a",))
            new_state = cache.pool_state_digest(
                ("candidate_a", "candidate_b")
            )
            self.assertNotEqual(old_state, new_state)

            old_baseline = cache.payload(
                ("base",),
                prediction_years=(2019, 2020, 2021),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
            )
            new_baseline = cache.payload(
                ("base",),
                prediction_years=(2019, 2020, 2021),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
            )
            self.assertEqual(old_baseline, new_baseline)

    def test_force_refresh_bypasses_an_exact_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(Path(directory), ("base",))
            first, first_hit, first_key = cache.get_or_compute(
                ("base",),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                compute=lambda: self.prediction,
            )
            refreshed, second_hit, second_key = cache.get_or_compute(
                ("base",),
                prediction_years=(2021,),
                label_column="ret_close_to_close",
                train_window_days=60,
                test_window_days=20,
                force_refresh=True,
                compute=lambda: self.prediction.assign(
                    factor=self.prediction["factor"] + 1
                ),
            )
            self.assertFalse(first_hit)
            self.assertFalse(second_hit)
            self.assertEqual(first_key, second_key)
            self.assertFalse(first["factor"].equals(refreshed["factor"]))


if __name__ == "__main__":
    unittest.main()
