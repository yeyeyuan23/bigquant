from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from train_unified_microstructure_v3_final import build_final_split


def test_build_final_split_keeps_one_full_label_gap() -> None:
    dates = pd.bdate_range("2024-12-23", periods=6)

    training, validation = build_final_split(dates, pd.Timestamp("2024-12-26"))

    assert dates[training[-1]] == pd.Timestamp("2024-12-26")
    assert dates[validation[0]] == pd.Timestamp("2024-12-30")
    assert int(training[-1]) == int(validation[0]) - 2


def test_build_final_split_requires_exact_cutoff_and_audit_dates() -> None:
    dates = pd.bdate_range("2024-12-23", periods=4)

    with pytest.raises(ValueError, match="not an available label date"):
        build_final_split(dates, pd.Timestamp("2024-12-28"))
    with pytest.raises(ValueError, match="at least two later label dates"):
        build_final_split(dates, pd.Timestamp("2024-12-26"))
