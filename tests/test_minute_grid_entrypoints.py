"""Current-minute packing and export must agree without relabeling old weights."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_models.microstructure import (
    MicrostructureModel,
    compact_packed_minutes,
)
from scripts.build_unified_m_raw_submission import build_sources


def test_compact_export_preserves_all_240_observations_and_rejects_data_loss():
    real = np.arange(2 * 240 * 17, dtype=np.float32).reshape(2, 240, 17)
    legacy = np.full((2, 242, 17), np.nan, dtype=np.float32)
    legacy[:, :240] = real
    fixed = np.full_like(legacy, np.nan)
    fixed[:, 1:121] = real[:, :120]
    fixed[:, 122:] = real[:, 120:]
    for values in (real, legacy, fixed):
        np.testing.assert_array_equal(compact_packed_minutes(values), real)
    with pytest.raises(ValueError, match="discard observed"):
        compact_packed_minutes(np.ones_like(legacy))
    legacy[0, 240, 0] = 3.0
    with pytest.raises(ValueError, match="discard observed"):
        compact_packed_minutes(legacy)
    with pytest.raises(ValueError, match="unexpected packed value shape"):
        compact_packed_minutes(real[:, :239])


def test_raw_export_uses_checkpoint_length_and_current_source(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path / "current.pt"
    model = MicrostructureModel(model_dim=8, kernels=(3,), tcn_blocks=1, max_minutes=240)
    model.save(checkpoint)
    snapshots = tmp_path / "sources"
    snapshots.mkdir()
    for name in ("base", "temporal", "microstructure"):
        (snapshots / f"alpha_models_{name}.py").write_text(
            (root / "src/alpha_models" / f"{name}.py").read_text()
        )
    sources, manifest = build_sources(
        checkpoint, snapshots,
        expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        training_commit="test", training_tree_dirty=False, checkpoint_block=0,
        training_start="2019-01-02", training_end="2023-12-28",
        training_protocol="test",
    )
    assert manifest["input_schema"]["max_minutes"] == 240
    for name, source in sources.items():
        (tmp_path / name).write_text(source)
        monkeypatch.delitem(sys.modules, Path(name).stem, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    import base64
    import io
    import zlib

    entry = importlib.import_module("unified_m_raw")
    restored = entry._load_frozen_model(torch, io, base64, zlib, torch.device("cpu"))
    assert restored.config.max_minutes == 240
    values = torch.randn(1, 3, 240, 17)
    masks = (torch.ones_like(values, dtype=torch.bool),
             torch.ones(1, 3, 240, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(restored.predict((values, *masks)), model.predict((values, *masks)))
    for name in sources:
        sys.modules.pop(Path(name).stem, None)


def test_current_raw_export_rejects_historical_checkpoint(tmp_path):
    checkpoint = tmp_path / "historical.pt"
    MicrostructureModel(max_minutes=242).save(checkpoint)
    with pytest.raises(ValueError, match="240-minute checkpoint"):
        build_sources(
            checkpoint, tmp_path,
            expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_commit="historical", training_tree_dirty=False, checkpoint_block=0,
            training_start="2019-01-02", training_end="2023-12-28",
            training_protocol="test",
        )
