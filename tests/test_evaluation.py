from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.evaluation import (
    LABEL_COLUMNS,
    build_return_labels,
    daily_prices_from_bar,
    evaluate_single_factor,
    rolling_elastic_net_scores,
)
from bigalpha2026.hf_pressure import main as hf_main
from bigalpha2026.search import enumerate_candidate_specs
from bigalpha2026.research import build_candidate_library
from bigalpha2026.synthetic import make_synthetic_datasources


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.datasources = make_synthetic_datasources(
            days=85,
            instruments=12,
            minutes_per_day=6,
        )
        universe = cls.datasources["bigalpha_2026_instruments"]
        cls.start = universe["date"].min()
        cls.end = universe["date"].max()
        cls.factor = hf_main(cls.datasources, cls.start, cls.end)
        cls.labels = build_return_labels(
            daily_prices_from_bar(cls.datasources["bigalpha_2026_stock_bar1m"])
        )

    def test_all_three_labels_and_a_item_proxies(self) -> None:
        self.assertEqual(
            list(self.labels.columns),
            ["date", "instrument", *LABEL_COLUMNS],
        )
        metrics = evaluate_single_factor(
            self.factor,
            self.labels,
            self.datasources["bigalpha_2026_exposure"],
        )
        self.assertEqual(set(metrics), set(LABEL_COLUMNS))
        self.assertIn("rank_ic_mean", metrics["ret_close_to_close"])
        self.assertIn("stress_ic_ir", metrics["ret_close_to_close"])

    def test_rolling_elastic_net_returns_stability_metrics(self) -> None:
        panel = self.factor.rename(columns={"factor": "candidate_a"})
        panel["candidate_b"] = -panel["candidate_a"] + np.random.default_rng(1).normal(
            0, 0.05, len(panel)
        )
        scores, weights = rolling_elastic_net_scores(
            panel,
            self.labels,
            ["candidate_a", "candidate_b"],
        )
        self.assertEqual(set(scores["factor"]), {"candidate_a", "candidate_b"})
        self.assertIn("nonzero_window_ratio", scores.columns)
        self.assertFalse(weights.empty)

    def test_search_space_contains_48_auditable_candidates(self) -> None:
        specs = enumerate_candidate_specs()
        self.assertEqual(len(specs), 48)
        self.assertEqual(len({spec.candidate_id for spec in specs}), 48)

    def test_candidate_library_materializes_both_families(self) -> None:
        specs = [
            enumerate_candidate_specs()[0],
            enumerate_candidate_specs()[24],
        ]
        small = make_synthetic_datasources(
            days=8,
            instruments=6,
            minutes_per_day=5,
        )
        universe = small["bigalpha_2026_instruments"]
        library = build_candidate_library(
            small,
            universe["date"].min(),
            universe["date"].max(),
            specs,
        )
        self.assertEqual(
            list(library.columns),
            ["date", "instrument", specs[0].candidate_id, specs[1].candidate_id],
        )
        self.assertFalse(library[[spec.candidate_id for spec in specs]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
