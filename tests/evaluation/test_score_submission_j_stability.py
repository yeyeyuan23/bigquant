import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_submission_j_stability import (
    CANDIDATE454_EXPECTED_COUNT,
    load_candidate454_reference_panel,
    score_individual_year,
)


class Candidate454SubmissionJTest(unittest.TestCase):
    def test_candidate454_loader_uses_every_manifest_factor(self):
        candidate_ids = tuple(
            f"TEST-{index:03d}"
            for index in range(CANDIDATE454_EXPECTED_COUNT)
        )
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            (store / "features" / "year=2024").mkdir(parents=True)
            manifest = {
                "candidate_count": CANDIDATE454_EXPECTED_COUNT,
                "candidate_ids": list(candidate_ids),
            }
            (store / "candidate454_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "instrument": ["A", "B"],
                    **{
                        candidate_id: np.array([index, index + 1], dtype=float)
                        for index, candidate_id in enumerate(candidate_ids)
                    },
                }
            )
            frame.to_parquet(
                store / "features" / "year=2024" / "part-2024.parquet",
                index=False,
            )

            panel, columns, metadata = load_candidate454_reference_panel(
                store, (2024,)
            )

            self.assertEqual(len(columns), CANDIDATE454_EXPECTED_COUNT)
            self.assertEqual(metadata["candidate_count"], 454)
            self.assertEqual(len(panel), 2)
            self.assertTrue(columns[0].startswith("candidate454__"))

    def test_each_route_gets_its_own_candidate454_fit(self):
        class FakeReference:
            reference_data_digest = "reference-digest"
            reference_columns = tuple(f"c{index}" for index in range(454))

            def __init__(self):
                self.calls = []

            def score(self, route):
                self.calls.append(route.copy())
                return {
                    "score_proxy": 0.7,
                    "a_proxy": 0.6,
                    "b_proxy": 0.75,
                    "b_model_score": 2.5,
                    "b_mean_abs_weight": 0.05,
                    "b_std_abs_weight": 0.02,
                    "b_nonzero_window_ratio": 1.0,
                }

        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        routes = {
            name: pd.DataFrame(
                {
                    "date": dates,
                    "instrument": ["A", "B"],
                    "factor": values,
                }
            )
            for name, values in {
                "route_a": [0.1, 0.2],
                "route_b": [0.3, 0.4],
            }.items()
        }
        reference = FakeReference()
        with tempfile.TemporaryDirectory() as directory:
            results = {
                name: score_individual_year(
                    year=2024,
                    version=name,
                    route=route,
                    route_source_digest=name,
                    score_reference=reference,
                    cache_dir=Path(directory),
                )
                for name, route in routes.items()
            }

        self.assertEqual(len(reference.calls), 2)
        self.assertEqual(
            [call["factor"].tolist() for call in reference.calls],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        self.assertEqual(results["route_a"]["reference_factor_count"], 454)
        self.assertEqual(results["route_b"]["candidate_route_count"], 1)
        self.assertEqual(results["route_a"]["score"]["b_model_score"], 2.5)


if __name__ == "__main__":
    unittest.main()
