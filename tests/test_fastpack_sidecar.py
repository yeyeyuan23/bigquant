"""E11 的 sidecar 连接：不给 sidecar 时行为必须不变，给了才追加 4 个通道。

为什么要锁住：sidecar 缺一天而静默少喂四个通道，比直接崩危险得多 ——
训练会照常跑完，结果却不是要测的东西，而且没有任何地方会报错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_models import MICROSTRUCTURE_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "finals_pre" / "common"))

from fastpack import SIDECAR_CHANNELS, load_microstructure_day_fast

DAY = pd.Timestamp("2024-06-03")
INSTRUMENTS = ("000001.SZ", "600000.SH")
MINUTES = 4


def _frame(columns: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for inst in INSTRUMENTS:
        for m in range(MINUTES):
            row = {"instrument": inst, "timestamp": DAY + pd.Timedelta(minutes=571 + m)}
            row.update({c: float(rng.normal()) for c in columns})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    part = root / "data" / f"trade_date={DAY.date()}"
    part.mkdir(parents=True)
    _frame(list(MICROSTRUCTURE_CHANNELS), seed=1).to_parquet(part / "part-000.parquet", index=False)
    return root


@pytest.fixture
def sidecar(tmp_path: Path) -> Path:
    root = tmp_path / "sidecar"
    part = root / f"trade_date={DAY.date()}"
    part.mkdir(parents=True)
    frame = _frame(list(SIDECAR_CHANNELS), seed=2)
    frame.loc[0, SIDECAR_CHANNELS[0]] = np.nan          # 缺失必须落成未观测
    frame.to_parquet(part / "part-000.parquet", index=False)
    return root


def test_without_sidecar_keeps_the_canonical_channel_count(store: Path) -> None:
    batch = load_microstructure_day_fast(store, DAY, INSTRUMENTS)
    assert batch.values.shape[-1] == len(MICROSTRUCTURE_CHANNELS)


def test_sidecar_appends_channels_and_leaves_the_first_seventeen_untouched(
    store: Path, sidecar: Path
) -> None:
    base = load_microstructure_day_fast(store, DAY, INSTRUMENTS)
    ext = load_microstructure_day_fast(store, DAY, INSTRUMENTS, sidecar=sidecar)

    n = len(MICROSTRUCTURE_CHANNELS)
    assert ext.values.shape[-1] == n + len(SIDECAR_CHANNELS)
    assert np.array_equal(base.values, ext.values[..., :n], equal_nan=True)
    assert np.array_equal(base.observed_mask, ext.observed_mask[..., :n])
    assert np.array_equal(base.minute_mask, ext.minute_mask)

    # sidecar 里那个 NaN 必须变成未观测，而不是被当成 0
    flat = ext.observed_mask[0, :, :MINUTES, n]
    assert flat.sum() == len(INSTRUMENTS) * MINUTES - 1


def test_missing_sidecar_partition_is_loud(store: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sidecar partition is missing"):
        load_microstructure_day_fast(store, DAY, INSTRUMENTS, sidecar=tmp_path / "nope")


def test_duplicate_sidecar_keys_are_rejected(store: Path, tmp_path: Path) -> None:
    root = tmp_path / "dup"
    part = root / f"trade_date={DAY.date()}"
    part.mkdir(parents=True)
    frame = _frame(list(SIDECAR_CHANNELS), seed=3)
    pd.concat([frame, frame.head(1)], ignore_index=True).to_parquet(
        part / "part-000.parquet", index=False
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        load_microstructure_day_fast(store, DAY, INSTRUMENTS, sidecar=root)
