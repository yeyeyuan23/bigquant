from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_unified_temporal.py"
SPEC = importlib.util.spec_from_file_location("evaluate_unified_temporal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
unified_temporal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(unified_temporal)


@pytest.mark.parametrize(
    ("half", "expected"),
    [
        (
            "H1",
            (
                pd.Timestamp("2022-12-31"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-06-30"),
            ),
        ),
        (
            "h2",
            (
                pd.Timestamp("2023-06-30"),
                pd.Timestamp("2023-07-01"),
                pd.Timestamp("2023-12-31"),
            ),
        ),
    ],
)
def test_fold_boundaries_are_causal_and_cover_each_half(
    half: str,
    expected: tuple[pd.Timestamp, ...],
) -> None:
    assert unified_temporal.fold_boundaries(2023, half) == expected


def test_fold_boundaries_reject_unknown_half() -> None:
    with pytest.raises(ValueError, match="unsupported validation half"):
        unified_temporal.fold_boundaries(2023, "q1")
