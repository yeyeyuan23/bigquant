from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from bigalpha2026.alpha_models import (
    MICROSTRUCTURE_CHANNELS,
    PRICE_COLUMNS,
    MicrostructureConfig,
    MicrostructureModel,
    MicrostructureNetwork,
    MicrostructureSourceConfig,
    ModelFactory,
    build_microstructure_features,
    canonicalize_microstructure_input,
    pack_microstructure_days,
    validate_instrument_map,
)
from bigalpha2026.alpha_models.microstructure import MicrostructureTCNBlock
from scripts.evaluate_unified_microstructure import (
    load_microstructure_day,
    prepare_label_panel,
    validate_micro_store,
)
from scripts.prepare_unified_microstructure_store import build_store


def raw_minutes() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-02 13:00",
            "2024-01-02 13:01",
        ]
    )
    rows = []
    for stock_index, instrument in enumerate(("000001.SZ", "000002.SZ")):
        for minute_index, timestamp in enumerate(timestamps):
            close = 10.0 + stock_index + minute_index * 0.01
            row: dict[str, object] = {
                "date": timestamp,
                "instrument": instrument,
                "open": close - 0.005,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "amount": 100_000 + minute_index * 1_000,
                "volume": 10_000 + minute_index * 100,
                "deal_number": 100 + minute_index,
            }
            for level in (1, 2, 3):
                row[f"ask_price{level}"] = close + level * 0.01
                row[f"bid_price{level}"] = close - level * 0.01
                row[f"ask_volume{level}"] = 80 + level * 10
                row[f"bid_volume{level}"] = 120 + level * 10
            rows.append(row)
    return pd.DataFrame(rows)


def small_config() -> MicrostructureConfig:
    return MicrostructureConfig(
        model_dim=16,
        max_minutes=8,
        kernels=(2, 3),
        tcn_blocks=2,
        tail_minutes=2,
        dropout=0.0,
    )


def compressed_e2e_minutes() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw_minutes()
    mapping = pd.DataFrame(
        {
            "instrument_id": [11, 22],
            "instrument": ["000001.SZ", "000002.SZ"],
        }
    )
    ids = {"000001.SZ": 11, "000002.SZ": 22}
    raw["instrument_id"] = raw.pop("instrument").map(ids)
    raw[list(PRICE_COLUMNS)] = raw[list(PRICE_COLUMNS)] * 100.0
    raw["amount"] *= 100.0
    return raw, mapping


def test_raw_feature_builder_sorts_and_resets_return_at_lunch() -> None:
    raw = raw_minutes().sample(frac=1.0, random_state=3)
    features = build_microstructure_features(raw)
    stock = features.loc[features["instrument"] == "000001.SZ"]
    assert stock["timestamp"].is_monotonic_increasing
    assert pd.isna(stock.iloc[0]["minute_log_return"])
    assert np.isfinite(stock.iloc[1]["minute_log_return"])
    assert pd.isna(stock.iloc[2]["minute_log_return"])
    assert np.isfinite(stock.iloc[3]["minute_log_return"])
    assert tuple(features.columns[-len(MICROSTRUCTURE_CHANNELS) :]) == MICROSTRUCTURE_CHANNELS


def test_raw_feature_builder_rejects_duplicate_minutes() -> None:
    raw = raw_minutes()
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate instrument/timestamp"):
        build_microstructure_features(duplicate)


def test_raw_feature_builder_rejects_missing_contract_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_microstructure_features(raw_minutes().drop(columns="bid_volume3"))


def test_e2e_profile_maps_ids_and_applies_certified_units() -> None:
    compressed, mapping = compressed_e2e_minutes()
    canonical, audit = canonicalize_microstructure_input(
        compressed,
        MicrostructureSourceConfig.for_profile("e2e_compressed"),
        instrument_map=mapping,
    )
    expected = raw_minutes().sort_values(["date", "instrument"], kind="stable")
    expected = expected.reset_index(drop=True)
    expected["instrument"] = expected["instrument"].astype("string")
    pd.testing.assert_series_equal(canonical["instrument"], expected["instrument"])
    np.testing.assert_allclose(canonical["close"], expected["close"])
    np.testing.assert_allclose(canonical["amount"], expected["amount"])
    assert audit["source_profile"] == "e2e_compressed"
    assert audit["unit_transform"]["price_divisor"] == 100.0
    assert audit["unit_transform"]["amount_divisor"] == 100.0


def test_e2e_profile_rejects_missing_mapping_and_unit_mismatch() -> None:
    compressed, mapping = compressed_e2e_minutes()
    config = MicrostructureSourceConfig.for_profile("e2e_compressed")
    with pytest.raises(ValueError, match="missing 1 source IDs"):
        canonicalize_microstructure_input(
            compressed,
            config,
            instrument_map=mapping.iloc[[0]],
        )
    broken = compressed.copy()
    book_prices = [column for column in PRICE_COLUMNS if "price" in column]
    broken[book_prices] /= 100.0
    with pytest.raises(ValueError, match="units are inconsistent"):
        canonicalize_microstructure_input(broken, config, instrument_map=mapping)


def test_instrument_map_must_be_one_to_one() -> None:
    mapping = pd.DataFrame(
        {
            "instrument_id": [1, 2],
            "instrument": ["000001.SZ", "000001.SZ"],
        }
    )
    with pytest.raises(ValueError, match="not one-to-one"):
        validate_instrument_map(mapping)


def test_future_day_does_not_change_previous_microstructure_features() -> None:
    first_day = raw_minutes()
    future_day = first_day.copy()
    future_day["date"] = pd.to_datetime(future_day["date"]) + pd.Timedelta(days=1)
    future_day["amount"] *= 100
    prefix = build_microstructure_features(first_day)
    full = build_microstructure_features(pd.concat([first_day, future_day], ignore_index=True))
    prior = full.loc[full["trade_date"] == pd.Timestamp("2024-01-02")].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix, prior)


def test_pack_microstructure_days_builds_masks_without_silent_truncation() -> None:
    features = build_microstructure_features(raw_minutes())
    batch = pack_microstructure_days(
        features,
        instruments=("000001.SZ", "000002.SZ", "MISSING"),
        max_minutes=8,
    )
    assert batch.values.shape == (1, 3, 8, len(MICROSTRUCTURE_CHANNELS))
    assert batch.minute_mask[0, :2].sum(axis=1).tolist() == [4, 4]
    assert batch.stock_mask.tolist() == [[True, True, False]]
    assert not batch.observed_mask[0, 0, 0, 0]
    with pytest.raises(ValueError, match="max_minutes=3"):
        pack_microstructure_days(features, max_minutes=3)


def test_microstructure_tcn_is_causal() -> None:
    torch.manual_seed(2)
    block = MicrostructureTCNBlock(4, (2, 3), 0.0).eval()
    values = torch.randn(1, 8, 4)
    changed = values.clone()
    changed[:, 5:] += 100.0
    mask = torch.ones(1, 8, dtype=torch.bool)
    torch.testing.assert_close(block(values, mask)[:, :5], block(changed, mask)[:, :5])


def test_microstructure_network_masks_padding_and_is_stock_permutation_equivariant() -> None:
    torch.manual_seed(5)
    network = MicrostructureNetwork(small_config()).eval()
    values = torch.randn(2, 5, 8, len(MICROSTRUCTURE_CHANNELS))
    observed = torch.rand_like(values) > 0.1
    values = values.masked_fill(~observed, float("nan"))
    minutes = torch.ones(2, 5, 8, dtype=torch.bool)
    stocks = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
    minutes = minutes & stocks.unsqueeze(-1)

    scores = network(values, observed, minutes, stocks)
    assert scores.shape == (2, 5)
    assert torch.isfinite(scores).all()
    assert torch.count_nonzero(scores.masked_select(~stocks)) == 0

    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = network(
        values[:, permutation],
        observed[:, permutation],
        minutes[:, permutation],
        stocks[:, permutation],
    )
    torch.testing.assert_close(permuted, scores[:, permutation], atol=1e-5, rtol=1e-5)


def test_microstructure_model_factory_and_backward() -> None:
    adapter = ModelFactory.create("unified_microstructure", {
        "model_dim": 16,
        "max_minutes": 8,
        "kernels": (2, 3),
        "tcn_blocks": 1,
        "tail_minutes": 2,
        "dropout": 0.0,
    })
    assert isinstance(adapter, MicrostructureModel)
    network = adapter.network.train()
    values = torch.randn(1, 4, 8, len(MICROSTRUCTURE_CHANNELS))
    observed = torch.ones_like(values, dtype=torch.bool)
    minutes = torch.ones(1, 4, 8, dtype=torch.bool)
    stocks = torch.ones(1, 4, dtype=torch.bool)
    network(values, observed, minutes, stocks).square().mean().backward()
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_microstructure_adapter_round_trip(tmp_path) -> None:
    config = {
        "model_dim": 16,
        "max_minutes": 8,
        "kernels": (2, 3),
        "tcn_blocks": 1,
        "tail_minutes": 2,
        "dropout": 0.0,
    }
    adapter = ModelFactory.create("unified_microstructure", config)
    values = torch.randn(1, 3, 8, len(MICROSTRUCTURE_CHANNELS))
    observed = torch.ones_like(values, dtype=torch.bool)
    minutes = torch.ones(1, 3, 8, dtype=torch.bool)
    stocks = torch.ones(1, 3, dtype=torch.bool)
    expected = adapter.predict((values, observed, minutes, stocks))
    path = tmp_path / "micro.pt"
    adapter.save(path)
    restored = MicrostructureModel.load(path)
    torch.testing.assert_close(restored.predict((values, observed, minutes, stocks)), expected)
    with pytest.raises(ValueError, match="must contain"):
        restored.predict((values, observed, stocks))


def test_prepare_store_and_day_loader_use_canonical_manifest(tmp_path) -> None:
    first = raw_minutes()
    second = first.copy()
    second["date"] = pd.to_datetime(second["date"]) + pd.Timedelta(days=1)
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)
    store = tmp_path / "store"
    manifest = build_store([first_path, second_path], store)
    assert manifest["trading_days"] == 2
    assert manifest["minute_rows"] == 16
    assert manifest["storage"] == {
        "engine": "polars",
        "value_dtype": "float32",
        "compression": "zstd",
        "compression_level": 3,
        "atomic_day_writes": True,
        "resumable": True,
    }
    progress = json.loads((store / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    partition = store / "data/trade_date=2024-01-02/part-000.parquet"
    parquet = pq.ParquetFile(partition)
    assert {
        parquet.metadata.row_group(0).column(index).compression
        for index in range(parquet.metadata.num_columns)
    } == {"ZSTD"}
    assert all(
        str(parquet.schema_arrow.field(channel).type) == "float"
        for channel in MICROSTRUCTURE_CHANNELS
    )

    stored = pd.read_parquet(partition)
    expected = build_microstructure_features(first).drop(columns="trade_date")
    pd.testing.assert_frame_equal(
        stored[["instrument", "timestamp"]],
        expected[["instrument", "timestamp"]],
        check_dtype=False,
    )
    np.testing.assert_allclose(
        stored[list(MICROSTRUCTURE_CHANNELS)],
        expected[list(MICROSTRUCTURE_CHANNELS)],
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    )
    validate_micro_store(store)
    batch = load_microstructure_day(
        store,
        pd.Timestamp("2024-01-02"),
        ("000001.SZ", "000002.SZ"),
        max_minutes=8,
    )
    assert batch is not None
    assert batch.stock_mask.tolist() == [[True, True]]
    assert load_microstructure_day(
        store,
        pd.Timestamp("2024-01-05"),
        ("000001.SZ",),
        max_minutes=8,
    ) is None


def test_prepare_store_resumes_from_completed_inputs(tmp_path) -> None:
    first = raw_minutes()
    second = first.copy()
    second["date"] = pd.to_datetime(second["date"]) + pd.Timedelta(days=1)
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)
    store = tmp_path / "store"

    initial = build_store([first_path], store)
    resumed = build_store([first_path, second_path], store, resume=True)

    assert initial["trading_days"] == 1
    assert resumed["trading_days"] == 2
    progress = json.loads((store / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert len(progress["completed_inputs"]) == 2
    with pytest.raises(FileExistsError, match="not empty"):
        build_store([first_path, second_path], store)


def test_prepare_store_records_e2e_mapping_and_unit_evidence(tmp_path) -> None:
    compressed, mapping = compressed_e2e_minutes()
    input_path = tmp_path / "e2e.parquet"
    mapping_path = tmp_path / "instrument_map.csv"
    compressed.to_parquet(input_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    store = tmp_path / "e2e_store"
    manifest = build_store(
        [input_path],
        store,
        source_config=MicrostructureSourceConfig.for_profile("e2e_compressed"),
        instrument_map_path=mapping_path,
    )
    assert manifest["schema_version"] == 2
    assert manifest["source_profile"] == "e2e_compressed"
    assert manifest["instrument_map"]["sha256"]
    assert manifest["input_files"][0]["audit"]["median_mid_to_close"] == pytest.approx(1.0)


def test_prepare_label_panel_ranks_each_day_and_rejects_duplicates() -> None:
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "instrument": ["A", "B", "C"],
            "ret_next_open_to_close": [-0.1, 0.0, 0.2],
        }
    )
    dates, targets = prepare_label_panel(labels)
    assert dates.tolist() == [pd.Timestamp("2024-01-02")]
    np.testing.assert_allclose(targets[dates[0]].to_numpy(), [-1 / 3, 1 / 3, 1.0])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_label_panel(pd.concat([labels, labels.iloc[[0]]], ignore_index=True))
