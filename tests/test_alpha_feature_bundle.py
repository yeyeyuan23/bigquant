from __future__ import annotations

import pandas as pd
import pytest

from bigalpha2026.alpha_models.feature_bundle import (
    CANDIDATE454,
    candidate_long_to_wide,
    get_feature_bundle,
)
from bigalpha2026.alpha_models.training_data import (
    load_candidate_feature_panel,
    panel_arrays,
)


def test_named_feature_bundles_have_expected_dimensions() -> None:
    assert CANDIDATE454.total_feature_count == 454
    assert get_feature_bundle("candidate454") is CANDIDATE454


def test_candidate_long_to_wide_preserves_requested_order() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "instrument": ["A", "A", "B", "B"],
            "candidate_id": ["HF-001", "PV-001", "HF-001", "PV-001"],
            "factor": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wide = candidate_long_to_wide(frame, candidate_ids=("PV-001", "HF-001"))
    assert list(wide.columns) == [
        "date",
        "instrument",
        "candidate__PV-001",
        "candidate__HF-001",
    ]


def test_panel_arrays_contains_only_candidate_inputs() -> None:
    candidate = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "instrument": ["A", "B"],
            "candidate__HF-001": [0.1, 0.2],
        }
    )
    labels = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["A", "B"],
            "ret_next_open_to_close": [-0.1, 0.1],
        }
    )
    panel = panel_arrays(candidate, labels)
    assert panel.candidate_values.shape == (1, 2, 1)
    assert panel.candidate_columns == ("candidate__HF-001",)


def test_panel_arrays_rejects_duplicate_candidate_keys() -> None:
    candidate = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["A", "A"],
            "candidate__HF-001": [0.1, 0.2],
        }
    )
    labels = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["A"],
            "ret_next_open_to_close": [0.1],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        panel_arrays(candidate, labels)


def test_load_candidate_feature_panel_supports_partitioned_wide_store(
    tmp_path,
) -> None:
    store = tmp_path / "features" / "year=2024"
    store.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "instrument": ["A", "A"],
            "HF-001": [0.1, float("nan")],
            "PV-001": [0.2, 0.3],
        }
    ).to_parquet(store / "part-2024.parquet", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"layout":"wide_partitioned","candidate_rows":{"HF-001":1,"PV-001":2}}',
        encoding="utf-8",
    )
    panel, ids = load_candidate_feature_panel(
        tmp_path / "features",
        manifest,
        start_date="2024-01-01",
        end_date="2024-12-31",
        expected_count=2,
    )
    assert ids == ("HF-001", "PV-001")
    assert list(panel.columns) == [
        "date",
        "instrument",
        "candidate__HF-001",
        "candidate__PV-001",
    ]
    assert pd.isna(panel.loc[1, "candidate__HF-001"])
