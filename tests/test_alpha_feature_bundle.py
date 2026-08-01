from __future__ import annotations

import pandas as pd
import pytest

from bigalpha2026.alpha_models.bar_features import (
    BAR1M_BASE_COLUMNS,
    build_bar1m_all156,
)
from bigalpha2026.alpha_models.feature_bundle import (
    ALL618,
    BAR156,
    CANDIDATE462,
    candidate_long_to_wide,
    get_feature_bundle,
    merge_feature_bundle,
)
from bigalpha2026.alpha_models.training_data import (
    load_candidate_feature_panel,
    panel_arrays,
)


def test_named_feature_bundles_have_expected_dimensions() -> None:
    assert BAR156.total_feature_count == 156
    assert CANDIDATE462.total_feature_count == 462
    assert ALL618.total_feature_count == 618
    assert get_feature_bundle("all618") is ALL618


def test_canonical_bar_builder_has_exactly_156_features() -> None:
    rows = []
    for instrument, scale in (("A", 1.0), ("B", 2.0)):
        for day in pd.date_range("2024-01-01", periods=5):
            row = {"date": day, "instrument": instrument}
            row.update(
                {
                    column: scale * (day.day + offset + 1)
                    for offset, column in enumerate(BAR1M_BASE_COLUMNS)
                }
            )
            rows.append(row)
    features, manifest = build_bar1m_all156(pd.DataFrame(rows))
    assert features.shape[1] == 158
    assert len(manifest) == 156
    assert "price_dispersion__raw" in features
    assert "realized_vol__raw" not in features


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


def test_merge_feature_bundle_rejects_duplicate_keys() -> None:
    bar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["A", "A"],
            "bar": [1, 2],
        }
    )
    candidate = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["A"],
            "candidate__X": [3],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        merge_feature_bundle(bar, candidate)


def test_panel_arrays_separates_bar_and_candidate_towers() -> None:
    bar = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "instrument": ["A", "B"],
            "bar_1": [1.0, 2.0],
            "bar_2": [3.0, 4.0],
        }
    )
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
    panel = panel_arrays(bar, labels, candidate)
    assert panel.bar_values.shape == (1, 2, 2)
    assert panel.candidate_values is not None
    assert panel.candidate_values.shape == (1, 2, 1)
    assert panel.bar_columns == ("bar_1", "bar_2")
    assert panel.candidate_columns == ("candidate__HF-001",)


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
