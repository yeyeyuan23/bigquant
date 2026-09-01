from __future__ import annotations

"""E5 raw-channel isolation contracts."""

import json

import pytest

torch = pytest.importorskip("torch")

from conftest import FINALS_PRE


@pytest.fixture(scope="module")
def e5_model(load_experiment_module):
    return load_experiment_module("e5_raw23_direct/model_raw23.py", "pre_e5_model_raw23")


@pytest.fixture(scope="module")
def e5_train(load_experiment_module):
    pytest.importorskip("pyarrow")
    return load_experiment_module("e5_raw23_direct/e5_train_autodl.py", "pre_e5_train")


def test_parquet_manifest_requires_exactly_40_by_242(e5_train, tmp_path, contract):
    expected = contract["experiments"]["E5"]
    manifest = {
        "channels": [f"channel_{index}" for index in range(expected["source_channels"])],
        "max_minutes": expected["max_minutes"],
    }
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    channels, minutes = e5_train.load_contract(tmp_path)
    assert len(channels) == expected["source_channels"]
    assert minutes == expected["max_minutes"]

    manifest["channels"].pop()
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected Parquet contract"):
        e5_train.load_contract(tmp_path)


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
