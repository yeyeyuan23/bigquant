from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "experiments/legacy_submissions/x_tree"


def _load_components_module():
    path = PACKAGE / "unified_candidate454_components.py"
    spec = importlib.util.spec_from_file_location("x_tree_bundle_components", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_read_columns_helper_is_defined(tmp_path: Path) -> None:
    components = _load_components_module()
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"date": ["2024-01-02"], "value": [1.0]}).to_parquet(
        path,
        index=False,
    )
    loaded = components.read_columns(path, ["value"])
    assert loaded.to_dict(orient="list") == {"value": [1.0]}
