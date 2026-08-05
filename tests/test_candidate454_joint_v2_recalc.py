from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_candidate454_joint_v2_recalc import (
    DETAIL_COLUMNS,
    FULL_RESULT_COLUMNS,
    SUMMARY_RESULT_COLUMNS,
    _write_reports,
    daily_rank,
)


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


def test_report_csvs_always_include_full_ab_detail_schema(tmp_path) -> None:
    summary = {
        "route": "route_a",
        "family": "test",
        "years": [2019, 2020, 2021, 2022, 2023, 2024],
        "J": 0.3,
        "A": 0.4,
        "B": 0.2571428571428572,
        **{column: 1.0 for column in DETAIL_COLUMNS},
    }
    _write_reports(
        tmp_path,
        {"protocol": "test_full_result_schema"},
        {},
        [{"name": "six_year", "summaries": [summary]}],
    )

    full = pd.read_csv(tmp_path / "joint_official_proxy_ab_details.csv")
    compact = pd.read_csv(tmp_path / "joint_official_proxy_summary.csv")

    assert tuple(full.columns) == FULL_RESULT_COLUMNS
    assert tuple(compact.columns) == SUMMARY_RESULT_COLUMNS
    assert not full.isna().any().any()
    assert not compact.isna().any().any()
