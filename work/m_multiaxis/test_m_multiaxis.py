from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_m_multiaxis import stability_loss
from unified_m_multiaxis_model import (
    AXIS_NAMES,
    AXIS_TOKEN_DIM,
    RAW_FIELDS,
    MultiAxisConfig,
    MultiAxisNetwork,
    _equal_event_segments,
    checkpoint_payload,
    load_checkpoint,
    pack_multiaxis_day,
)


def _stats() -> dict[str, object]:
    return {
        "fields": list(RAW_FIELDS),
        "mean": [0.0] * len(RAW_FIELDS),
        "std": [1.0] * len(RAW_FIELDS),
        "count": [100] * len(RAW_FIELDS),
        "fit_start": "2019-01-02",
        "fit_end": "2024-12-26",
    }


def _frame(stocks: int = 3, minutes: int = 240) -> pd.DataFrame:
    rows = []
    for key in range(stocks):
        for minute in range(minutes):
            values = np.sin(np.arange(len(RAW_FIELDS)) * 0.1 + minute * 0.03 + key)
            values[RAW_FIELDS.index("close")] = 4.0 + 0.001 * minute + 0.02 * key
            values[RAW_FIELDS.index("amount")] = np.log1p(1000 + (minute + 1) ** 2 + key)
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-02 09:31") + pd.Timedelta(minutes=minute + (90 if minute >= 120 else 0)),
                    "key": key,
                    **dict(zip(RAW_FIELDS, values, strict=True)),
                }
            )
    return pd.DataFrame(rows)


def test_event_segments_are_chronological_contiguous_and_exhaustive() -> None:
    weights = np.geomspace(1.0, 100.0, 240)
    segments = _equal_event_segments(weights, 24)
    assert len(segments) == 24
    assert segments[0][0] == 0
    assert segments[-1][1] == len(weights)
    assert all(left < right for left, right in segments)
    assert all(first[1] == second[0] for first, second in pairwise(segments))


def test_pack_contract_and_determinism() -> None:
    frame = _frame()
    first = pack_multiaxis_day(frame, keys=(0, 1, 2), stats=_stats())
    second = pack_multiaxis_day(frame.sample(frac=1.0, random_state=7), keys=(0, 1, 2), stats=_stats())
    assert first.tail_values.shape == (1, 3, 60, len(RAW_FIELDS))
    assert first.axis_values.shape == (1, 3, len(AXIS_NAMES), 24, AXIS_TOKEN_DIM)
    assert first.tail_mask.all() and first.axis_mask.all() and first.stock_mask.all()
    np.testing.assert_allclose(first.tail_values, second.tail_values)
    np.testing.assert_allclose(first.axis_values, second.axis_values)


def test_axes_have_distinct_variable_duration_metadata() -> None:
    batch = pack_multiaxis_day(_frame(stocks=1), keys=(0,), stats=_stats())
    duration_offset = len(RAW_FIELDS) * 3
    time_duration = batch.axis_values[0, 0, 0, :, duration_offset]
    flow_duration = batch.axis_values[0, 0, 2, :, duration_offset]
    assert np.unique(time_duration).size <= 2
    assert np.unique(flow_duration).size > 2


def test_network_forward_checkpoint_and_nonconstant_output() -> None:
    torch.manual_seed(11)
    batch = pack_multiaxis_day(_frame(), keys=(0, 1, 2), stats=_stats())
    model = MultiAxisNetwork(MultiAxisConfig(dropout=0.0)).eval()
    args = (
        torch.from_numpy(batch.tail_values),
        torch.from_numpy(batch.tail_mask),
        torch.from_numpy(batch.axis_values),
        torch.from_numpy(batch.axis_mask),
        torch.from_numpy(batch.stock_mask),
    )
    with torch.inference_mode():
        output = model(*args)
    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()
    assert torch.unique(output).numel() > 1
    payload = checkpoint_payload(model, stats=_stats(), seed=11, training={"smoke": True})
    loaded, _ = load_checkpoint(payload, device=torch.device("cpu"))
    with torch.inference_mode():
        replay = loaded(*args)
    torch.testing.assert_close(output, replay)


def test_stability_loss_backpropagates_mean_and_dispersion() -> None:
    values = torch.tensor([0.01, 0.03, -0.02, 0.04], requires_grad=True)
    loss = stability_loss(values, 0.15)
    loss.backward()
    assert torch.isfinite(loss)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
