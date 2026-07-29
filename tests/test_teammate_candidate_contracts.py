from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[1] / "src" / "bigalpha2026" / "candidates"
TEAMMATE_RANGES = {
    "composite": {f"int_{index:03d}" for index in range(4, 18)},
    "hf": {f"hf_{index:03d}" for index in range(5, 79)},
    "ob": {"ob_006"},
    "pv": {f"pv_{index:03d}" for index in range(24, 45)},
}


def _module_names() -> list[str]:
    modules: list[str] = []
    for family, stems in TEAMMATE_RANGES.items():
        for source in sorted((CANDIDATE_ROOT / family).glob("*.py")):
            if source.stem in stems:
                modules.append(f"bigalpha2026.candidates.{family}.{source.stem}")
    return modules


TEAMMATE_MODULES = _module_names()


def _daily_panel(required_columns: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2022-01-03", periods=4)
    instruments = ["A", "B", "C", "D"]
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for instrument_index, instrument in enumerate(instruments):
            row: dict[str, object] = {"date": date, "instrument": instrument}
            for column_index, column in enumerate(sorted(required_columns)):
                row[column] = (
                    10.0 * (column_index + 1)
                    + 1.5 * date_index
                    + float(instrument_index)
                )
            rows.append(row)
    daily = pd.DataFrame(rows)
    pool = daily[["date", "instrument"]].copy()
    return daily, pool


def _required_feature_columns(module: object) -> set[str]:
    if hasattr(module, "COMPONENT_COLUMN"):
        return {str(module.COMPONENT_COLUMN)}
    if hasattr(module, "MEMBERS"):
        return {str(column) for column in module.MEMBERS}
    if hasattr(module, "SPREAD_COLUMN"):
        return {
            str(module.SPREAD_COLUMN),
            "micro_snapshot_available",
            "valid_snapshot_count",
        }
    raise AssertionError(f"{module.CANDIDATE_ID} does not declare feature columns")


@pytest.mark.parametrize("module_name", TEAMMATE_MODULES)
def test_teammate_candidate_daily_builder_contract(module_name: str) -> None:
    module = importlib.import_module(module_name)
    candidate_id = module_name.rsplit(".", maxsplit=1)[-1].upper().replace("_", "-")
    assert module.CANDIDATE_ID == candidate_id
    assert tuple(module.OUTPUT_COLUMNS) == ("date", "instrument", "factor")
    assert isinstance(module.INCLUDE_IN_J_BASELINE, bool)
    assert not hasattr(module, "INCLUDE_IN_SELF_LIBRARY")
    assert module.SEMANTIC_CLASS in {"LATENT_COMPONENT", "ANCHOR_COMPONENT"}

    required_columns = _required_feature_columns(module)
    daily, pool = _daily_panel(required_columns)
    builders = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("build_") and name.endswith("_factor_from_daily")
    ]
    assert len(builders) == 1

    result = builders[0](daily, pool)

    assert list(result.columns) == ["date", "instrument", "factor"]
    assert len(result) == len(pool)
    assert not result.duplicated(["date", "instrument"]).any()
    assert np.isfinite(result["factor"]).all()
    assert result["factor"].abs().max() <= 1.0
