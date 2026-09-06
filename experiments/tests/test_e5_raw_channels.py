from __future__ import annotations

"""E5 raw-channel isolation contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from conftest import FINALS_PRE, path_is_mounted

from alpha_models import MICROSTRUCTURE_CHANNELS, build_microstructure_features


@pytest.fixture(scope="module")
def e5_model(load_experiment_module):
    return load_experiment_module("e5_raw23_direct/model_raw23.py", "pre_e5_model_raw23")


@pytest.fixture(scope="module")
def e5_train(load_experiment_module):
    pytest.importorskip("pyarrow")
    return load_experiment_module("e5_raw23_direct/e5_train_autodl.py", "pre_e5_train")


def test_parquet_manifest_preserves_legacy_source_contract(e5_train, tmp_path, contract):
    expected = contract["experiments"]["E5"]
    manifest = {
        "channels": expected["channels"],
        "max_minutes": expected["max_minutes"],
    }
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    channels, minutes = e5_train.load_contract(tmp_path)
    assert len(channels) == expected["source_channels"]
    assert minutes == expected["max_minutes"]

    manifest["channels"] = manifest["channels"].copy()
    manifest["channels"][18], manifest["channels"][19] = (
        manifest["channels"][19],
        manifest["channels"][18],
    )
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected Parquet contract"):
        e5_train.load_contract(tmp_path)


def test_all_40_channel_names_and_order_match_the_actual_export(e5_train, contract):
    expected = tuple(contract["experiments"]["E5"]["channels"])
    assert e5_train.E5_RAW40_CHANNELS == expected
    assert expected[:17] == MICROSTRUCTURE_CHANNELS
    assert len(expected) == len(set(expected)) == 40


def test_actual_autodl_e5_store_matches_and_loads_the_40_channel_contract(e5_train):
    store = Path("/root/bigquant_private_data/e5_raw40_2023_2024_parquet")
    if not path_is_mounted(store / "export_manifest.json"):
        pytest.skip("AutoDL E5 store is not mounted in this environment")
    channels, max_minutes = e5_train.load_contract(store)
    first = min(store.glob("day=*.parquet"))
    day = pd.Timestamp(first.stem.removeprefix("day="))
    batch = e5_train.load_day(day, store=store, channels=channels, max_minutes=max_minutes)
    assert batch is not None
    instruments, values, observed, minute_mask, stock_mask = batch
    assert values.shape == (1000, 240, 40)
    assert observed.shape == values.shape
    assert len(instruments) == 1000
    assert minute_mask.shape == stock_mask.shape + (240,)


def test_unreadable_optional_mount_is_skipped_instead_of_failing(monkeypatch):
    def denied(_path):
        raise PermissionError("hosted runner cannot inspect /root")

    monkeypatch.setattr(Path, "exists", denied)
    assert path_is_mounted(Path("/root/autodl-tmp")) is False


def test_all_17_engineered_channel_formulas_against_independent_values():
    rows = []
    for minute, close in enumerate((10.0, 10.2)):
        row = {
            "date": pd.Timestamp("2024-06-03 09:31") + pd.Timedelta(minutes=minute),
            "instrument": "000001.SZ",
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.10,
            "close": close,
            "amount": 1000.0 + 100.0 * minute,
            "volume": 100.0 + 10.0 * minute,
            "deal_number": 10.0 + minute,
        }
        for level in (1, 2, 3):
            row[f"ask_price{level}"] = close + 0.01 * level
            row[f"bid_price{level}"] = close - 0.01 * level
            row[f"ask_volume{level}"] = 10.0 * level
            row[f"bid_volume{level}"] = 20.0 * level
        rows.append(row)
    actual = build_microstructure_features(pd.DataFrame(rows)).iloc[1]

    close = 10.2
    high, low = 10.4, 10.1
    ask1, bid1 = 10.21, 10.19
    ask_volumes = np.array([10.0, 20.0, 30.0])
    bid_volumes = np.array([20.0, 40.0, 60.0])
    mid = (ask1 + bid1) / 2.0
    spread = ask1 - bid1
    microprice = (ask1 * bid_volumes[0] + bid1 * ask_volumes[0]) / (bid_volumes[0] + ask_volumes[0])
    phase = 2.0 * np.pi * (572 - 570) / 330.0
    expected = {
        "minute_log_return": np.log(10.2 / 10.0),
        "bar_range": (high - low) / close,
        "close_location": (close - low) / (high - low) - 0.5,
        "relative_spread": spread / mid,
        "microprice_gap": (microprice - mid) / spread,
        "depth_imbalance_l1": (bid_volumes[0] - ask_volumes[0]) / (bid_volumes[0] + ask_volumes[0]),
        "depth_imbalance_l3": (bid_volumes.sum() - ask_volumes.sum())
        / (bid_volumes.sum() + ask_volumes.sum()),
        "depth_shape": (bid_volumes[0] + bid_volumes[1]) / bid_volumes.sum()
        - (ask_volumes[0] + ask_volumes[1]) / ask_volumes.sum(),
        "log_amount": np.log1p(1100.0),
        "log_volume": np.log1p(110.0),
        "log_deal_number": np.log1p(11.0),
        "log_amount_per_deal": np.log1p(1100.0 / 11.0),
        "log_volume_per_deal": np.log1p(110.0 / 11.0),
        "signed_log_amount": np.log1p(1100.0),
        "time_sin": np.sin(phase),
        "time_cos": np.cos(phase),
        "pm_session": 0.0,
    }
    assert tuple(expected) == MICROSTRUCTURE_CHANNELS
    np.testing.assert_allclose(
        actual[list(MICROSTRUCTURE_CHANNELS)].to_numpy(float),
        np.fromiter(expected.values(), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_baseline_contract_keeps_exactly_the_original_first_17(e5_model, contract):
    expected = contract["experiments"]["E5"]
    config = e5_model.ProgressiveConfig(
        input_dim=expected["source_channels"],
        keep_channels=tuple(expected["baseline_indices"]),
    )
    assert config.input_dim == expected["baseline_channels"]
    assert config.keep_channels == tuple(range(17))
    assert e5_model.ProgressiveConfig(input_dim=40).input_dim == 40


def test_launcher_pairs_every_seed_with_both_arms(contract):
    source = (FINALS_PRE / "e5_raw23_direct/run_e5_autodl.sh").read_text(encoding="utf-8")
    seeds = " ".join(str(seed) for seed in contract["global"]["seeds"])
    assert f"for seed in {seeds}; do" in source
    assert "for arm in baseline17 raw40; do" in source
    assert "--epochs 3" in source
    assert "--device cuda" in source


def test_every_seed_gives_both_e5_arms_the_same_shuffle(e5_train, contract):
    days = list(pd.bdate_range("2023-01-03", periods=40))
    for seed in contract["global"]["seeds"]:
        baseline = e5_train.training_day_plan(days, seed, epochs=3)
        raw40 = e5_train.training_day_plan(days, seed, epochs=3)
        assert baseline == raw40
        with pytest.raises(AssertionError):
            assert baseline == e5_train.training_day_plan(days, seed + 1, epochs=3)


def test_extra_23_channels_cannot_leak_into_baseline_output(e5_model):
    torch.manual_seed(7)
    config = e5_model.ProgressiveConfig(
        input_dim=40,
        keep_channels=tuple(range(17)),
        model_dim=8,
        max_minutes=8,
        kernels=(3,),
        tcn_blocks=1,
        tail_minutes=4,
        dropout=0.0,
    )
    model = e5_model.ProgressiveNetwork(config).eval()
    values = torch.randn(1, 5, 8, 40)
    changed = values.clone()
    changed[..., 17:] = torch.randn_like(changed[..., 17:]) * 10_000.0
    observed = torch.ones_like(values, dtype=torch.bool)
    minutes = torch.ones(1, 5, 8, dtype=torch.bool)
    stocks = torch.ones(1, 5, dtype=torch.bool)
    with torch.inference_mode():
        before = model(values, observed, minutes, stocks)
        after = model(changed, observed, minutes, stocks)
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
