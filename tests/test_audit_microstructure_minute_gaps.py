from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")
SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_microstructure_minute_gaps.py"
SPEC = importlib.util.spec_from_file_location("audit_microstructure_minute_gaps", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit_dataset = MODULE.audit_dataset


def test_minute_gap_audit_separates_internal_and_edge_gaps(tmp_path: Path) -> None:
    day = pd.Timestamp("2024-01-02")
    schedule = [
        *pd.date_range("09:31", "11:30", freq="min").strftime("%H:%M"),
        *pd.date_range("13:01", "15:00", freq="min").strftime("%H:%M"),
    ]
    rows: list[dict[str, object]] = []
    for minute in schedule:
        rows.append({"date": pd.Timestamp(f"{day.date()} {minute}"), "instrument": "FULL"})
    missing = {"09:45", "15:00"}
    for minute in schedule:
        if minute not in missing:
            rows.append({"date": pd.Timestamp(f"{day.date()} {minute}"), "instrument": "GAPPED"})
    rows.append(rows[-1].copy())
    source = tmp_path / "minutes.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    result = audit_dataset(
        [source],
        output_dir=tmp_path / "audit",
        dataset_name="synthetic",
    )

    assert result["stock_days"] == 2
    assert result["stock_days_with_missing_rows"] == 1
    assert result["missing_minute_rows"] == 2
    assert result["internal_missing_rows"] == 1
    assert result["trailing_missing_rows"] == 1
    assert result["duplicate_minute_rows"] == 1
    assert result["out_of_schedule_rows"] == 0
