from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


final_train = load_script("train_unified_final_checkpoint.py")
period_score = load_script("score_route_periods.py")

sys_path = str(ROOT / "src")
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from alpha_models.temporal import (
    CandidateTemporalConfig,
    CandidateTemporalNetwork,
    temporal_history_indices,
)


def test_final_training_indices_apply_lookback_eligibility_then_stride() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    targets = np.full((8, 3), np.nan)
    targets[2:, :2] = [[0.1, -0.1]] * 6

    indices, skipped = final_train.select_training_indices(
        dates,
        targets,
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-08"),
        lookback=3,
        stride=2,
    )

    assert indices == [2, 4, 6]
    assert skipped == 0


def test_final_checkpoint_names_keep_x_and_t_separate() -> None:
    assert final_train.final_checkpoint_name("mlp") == "unified_x_final_checkpoint.pt"
    assert final_train.final_checkpoint_name("temporal") == "unified_t_final_checkpoint.pt"
    with pytest.raises(ValueError, match="unsupported model"):
        final_train.final_checkpoint_name("micro")


@pytest.mark.parametrize(
    ("half", "expected"),
    [
        ("h1", (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))),
        ("H2", (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"))),
    ],
)
def test_period_bounds_are_fixed_calendar_halves(half, expected) -> None:
    assert period_score.period_bounds(2024, half) == expected


def test_temporal_history_modes_are_causal_and_keep_current_date() -> None:
    dense = temporal_history_indices(60, "dense").tolist()
    stride2 = temporal_history_indices(60, "stride2").tolist()
    multiscale = temporal_history_indices(60, "multiscale").tolist()

    assert dense == list(range(60))
    assert len(stride2) == 30
    assert len(multiscale) == 12
    assert stride2[-1] == multiscale[-1] == 59
    assert multiscale == sorted(set(multiscale))


@pytest.mark.parametrize("history_mode", ["dense", "stride2", "multiscale"])
def test_temporal_network_history_modes_produce_finite_scores(history_mode) -> None:
    torch = pytest.importorskip("torch")
    config = CandidateTemporalConfig(
        input_dim=8,
        model_dim=16,
        lookback=60,
        kernels=(3, 5),
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        dropout=0.0,
        history_mode=history_mode,
    )
    model = CandidateTemporalNetwork(config).eval()
    values = torch.randn(1, 4, 60, 8)
    observed = torch.ones_like(values, dtype=torch.bool)
    stocks = torch.ones(1, 4, dtype=torch.bool)

    with torch.inference_mode():
        scores = model(values, observed, stocks)

    assert scores.shape == (1, 4)
    assert torch.isfinite(scores).all()
