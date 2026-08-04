from __future__ import annotations

import pandas as pd
import pytest
import torch

from bigalpha2026.alpha_models.microstructure_v2 import (
    DEEP_BOOK_CONTEXT_CHANNELS,
    DYNAMIC_MICROSTRUCTURE_CHANNELS,
    MICROSTRUCTURE_V2_BASE_CHANNELS,
    MICROSTRUCTURE_V2_CHANNELS,
    GatedMicrostructureNetwork,
    MicrostructureV2Config,
    build_microstructure_v2_features,
    pack_microstructure_v2_days,
)
from scripts.evaluate_unified_microstructure_v2 import (
    validate_deep_book_prediction_coverage,
    validate_micro_store,
)
from scripts.prepare_unified_microstructure_store import build_store as build_v1_store
from scripts.prepare_unified_microstructure_v2_store import build_store
from scripts.upgrade_microstructure_v2_store import upgrade_store


def _minute_frame(*, levels: int = 5, days: int = 2) -> pd.DataFrame:
    instruments = ("000001.SZ", "000002.SZ", "000003.SZ")
    rows: list[dict[str, object]] = []
    for day_index, day in enumerate(pd.bdate_range("2024-01-02", periods=days)):
        times = pd.date_range(f"{day.date()} 09:31", periods=20, freq="min").append(
            pd.date_range(f"{day.date()} 13:01", periods=20, freq="min")
        )
        for stock_index, instrument in enumerate(instruments):
            for minute_index, timestamp in enumerate(times):
                mid = 10.0 + stock_index + day_index * 0.01 + minute_index * 0.0005
                row: dict[str, object] = {
                    "date": timestamp,
                    "instrument": instrument,
                    "open": mid - 0.001,
                    "high": mid + 0.003,
                    "low": mid - 0.003,
                    "close": mid + 0.001,
                    "amount": 100_000 + minute_index * 100,
                    "volume": 10_000 + minute_index * 10,
                    "deal_number": 100 + minute_index,
                }
                for level in range(1, levels + 1):
                    row[f"bid_price{level}"] = mid - level * 0.001
                    row[f"ask_price{level}"] = mid + level * 0.001
                    row[f"bid_volume{level}"] = (
                        1_000 + stock_index * 100 + minute_index * (level + 1 + stock_index)
                    )
                    row[f"ask_volume{level}"] = (
                        950 + (2 - stock_index) * 80 + minute_index * (level + 2 - stock_index)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _pool(raw: pd.DataFrame) -> pd.DataFrame:
    output = raw[["date", "instrument"]].copy()
    output["date"] = pd.to_datetime(output["date"]).dt.normalize()
    return output.drop_duplicates().sort_values(["date", "instrument"])


def _deep_context(raw: pd.DataFrame) -> pd.DataFrame:
    context = _pool(raw)
    stock_code = context["instrument"].str.slice(0, 6).astype(int)
    day_number = context.groupby("instrument", sort=False).cumcount()
    context["full_five_levels_rate"] = 0.95
    context["full_day_depth_imbalance_median"] = (
        0.03 + stock_code.mod(7) * 0.002 + day_number * 0.001
    )
    context["tail_60_bid_depth_imbalance_median"] = (
        0.05 + stock_code.mod(5) * 0.003 + day_number * 0.001
    )
    context["negative_mid_shock_q10_bid_depth_recovery_5m_median"] = (
        -0.1 + stock_code.mod(3) * 0.05
    )
    context["tail_60_shape_sign_consistency"] = 0.4 + stock_code.mod(4) * 0.1
    return context


def test_v3_contract_includes_explicit_l4_l5_context_channels() -> None:
    assert len(DEEP_BOOK_CONTEXT_CHANNELS) == 4
    assert set(DEEP_BOOK_CONTEXT_CHANNELS).issubset(MICROSTRUCTURE_V2_CHANNELS)
    assert all(name.startswith("l5_") for name in DEEP_BOOK_CONTEXT_CHANNELS)


def test_v2_features_are_session_safe_and_prefix_invariant() -> None:
    raw = _minute_frame(levels=3)
    first_day = raw.loc[pd.to_datetime(raw["date"]).dt.normalize().eq("2024-01-02")]
    prefix = build_microstructure_v2_features(first_day, _deep_context(first_day))
    full = build_microstructure_v2_features(raw, _deep_context(raw))
    prior = full.loc[full["trade_date"].eq(pd.Timestamp("2024-01-02"))]
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), prior.reset_index(drop=True))
    assert full[list(DEEP_BOOK_CONTEXT_CHANNELS)].notna().all().all()
    deep_unique = full.groupby(["trade_date", "instrument"], sort=False)[
        list(DEEP_BOOK_CONTEXT_CHANNELS)
    ].nunique()
    assert deep_unique.le(1).all().all()

    first_rows = full.groupby(
        ["trade_date", "instrument", full["timestamp"].dt.hour.ge(12)], sort=False
    ).head(1)
    delta_columns = [name for name in DYNAMIC_MICROSTRUCTURE_CHANNELS if name.startswith("delta_")]
    assert first_rows[delta_columns].isna().all().all()


def test_v3_masks_unavailable_five_level_context() -> None:
    raw = _minute_frame(levels=3, days=1)
    context = _deep_context(raw)
    context.loc[context["instrument"].eq("000001.SZ"), "full_five_levels_rate"] = 0.0
    features = build_microstructure_v2_features(raw, context)
    unavailable = features["instrument"].eq("000001.SZ")
    assert features.loc[unavailable, list(DEEP_BOOK_CONTEXT_CHANNELS)].isna().all().all()


def test_v3_rejects_missing_prediction_period_five_level_coverage() -> None:
    metadata = {
        "files": [
            {"year": 2023, "five_level_coverage": 0.98},
            {"year": 2024, "five_level_coverage": 0.0},
        ]
    }
    with pytest.raises(RuntimeError, match="prediction-period five-level coverage"):
        validate_deep_book_prediction_coverage(
            metadata,
            prediction_years=(2024,),
            minimum=0.5,
        )
    validate_deep_book_prediction_coverage(
        metadata,
        prediction_years=(2023,),
        minimum=0.5,
    )


def test_v2_pack_and_gated_network_backward() -> None:
    raw = _minute_frame(levels=3, days=1)
    features = build_microstructure_v2_features(raw, _deep_context(raw))
    batch = pack_microstructure_v2_days(features, max_minutes=40)
    assert batch.values.shape == (1, 3, 40, len(MICROSTRUCTURE_V2_CHANNELS))
    config = MicrostructureV2Config(
        model_dim=16,
        max_minutes=40,
        kernels=(2, 5),
        tcn_blocks=1,
        tail_minutes=10,
        dropout=0.0,
    )
    network = GatedMicrostructureNetwork(config).train()
    values = torch.from_numpy(batch.values)
    observed = torch.from_numpy(batch.observed_mask)
    minutes = torch.from_numpy(batch.minute_mask)
    stocks = torch.from_numpy(batch.stock_mask)
    scores = network(values, observed, minutes, stocks)
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores).all()
    scores.square().mean().backward()
    assert any(parameter.grad is not None for parameter in network.path_gate.parameters())


def test_v2_store_has_separate_schema_and_l1_l3_source_contract(tmp_path) -> None:
    source = tmp_path / "bar1m.parquet"
    _minute_frame(levels=3, days=1).to_parquet(source, index=False)
    store = tmp_path / "m_v2_store"
    manifest = build_store([source], store)
    assert manifest["schema_version"] == 3
    assert manifest["source_contract"].endswith("first_3_book_levels_dynamic_v2")
    assert tuple(manifest["channels"]) == MICROSTRUCTURE_V2_BASE_CHANNELS
    assert validate_micro_store(store)["schema_sha256"] == manifest["schema_sha256"]


def test_v1_store_can_be_upgraded_without_raw_recomputation(tmp_path) -> None:
    source = tmp_path / "bar1m.parquet"
    _minute_frame(levels=3, days=1).to_parquet(source, index=False)
    v1_store = tmp_path / "m_v1_store"
    v1_manifest = build_v1_store([source], v1_store)
    v2_store = tmp_path / "m_v2_store"
    v2_manifest = upgrade_store(v1_store, v2_store)
    assert v2_manifest["schema_version"] == 3
    assert v2_manifest["parent_store"]["schema_sha256"] == v1_manifest["schema_sha256"]
    assert v2_manifest["minute_rows"] == v1_manifest["minute_rows"]
    assert validate_micro_store(v2_store)["channels"] == list(
        MICROSTRUCTURE_V2_BASE_CHANNELS
    )
