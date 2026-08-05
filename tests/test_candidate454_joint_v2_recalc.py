from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_candidate454_joint_v2_recalc import daily_rank


def test_daily_rank_uses_per_day_average_ranks() -> None:
    dates = pd.Series(
        [
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ]
    )
    values = pd.Series([3.0, 1.0, 2.0, 10.0, 10.0])

    ranked = daily_rank(values, dates)

    assert ranked.tolist() == pytest.approx(
        [1.0, -1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5]
    )
