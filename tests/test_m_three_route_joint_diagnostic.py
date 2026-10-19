import pandas as pd
import pytest

from scripts.score_m_three_route_joint_diagnostic import (
    EXPECTED_ROUTE_NAMES,
    score_three_routes,
)


class FakeReference:
    def __init__(self):
        self.calls = []

    def score_joint_routes(self, routes):
        self.calls.append(tuple(routes))
        return {
            name: {
                "score_proxy": 0.8,
                "a_proxy": 0.7,
                "b_proxy": 0.85,
                "b_model_score": 2.0,
                "b_mean_abs_weight": 0.04,
                "b_std_abs_weight": 0.02,
                "b_nonzero_window_ratio": 1.0,
            }
            for name in routes
        }


def test_three_route_diagnostic_is_one_explicit_joint_fit():
    reference = FakeReference()
    routes = {
        name: pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "instrument": ["A"],
                "factor": [index / 10],
            }
        )
        for index, name in enumerate(EXPECTED_ROUTE_NAMES)
    }

    scores = score_three_routes(reference, routes)

    assert reference.calls == [EXPECTED_ROUTE_NAMES]
    assert set(scores) == set(EXPECTED_ROUTE_NAMES)


def test_three_route_diagnostic_rejects_seed_level_routes():
    reference = FakeReference()
    routes = {
        "old_m": pd.DataFrame(),
        "m_dynamic_seed_1": pd.DataFrame(),
        "m_l5_ensemble": pd.DataFrame(),
    }

    with pytest.raises(ValueError, match="requires ordered routes"):
        score_three_routes(reference, routes)
