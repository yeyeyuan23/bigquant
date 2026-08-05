from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import zlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submissions/x_tree"
MANIFEST = ROOT / "submissions/unified_x_tree_manifest.json"
CHECKPOINT = (
    ROOT
    / "reports/unified_alpha_fusion_suite_20260801_full_experts_v1/lightgbm"
    / "unified_lightgbm_2024_h2_checkpoint.txt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_weights_module():
    path = PACKAGE / "unified_x_tree_weights.py"
    spec = importlib.util.spec_from_file_location("x_tree_bundle_weights", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_components_module():
    path = PACKAGE / "unified_candidate454_components.py"
    spec = importlib.util.spec_from_file_location("x_tree_bundle_components", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_x_tree_upload_bundle_matches_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    declared = {item["name"]: item for item in manifest["files"]}
    assert len(declared) == manifest["upload_file_count"] == 6
    assert set(declared) == {
        "unified_candidate454_components.py",
        "unified_candidate454_direct.py",
        "unified_candidate454_gtja.py",
        "unified_x_tree.ipynb",
        "unified_x_tree.py",
        "unified_x_tree_weights.py",
    }
    for name, metadata in declared.items():
        path = PACKAGE / name
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]


def test_x_tree_embeds_selected_checkpoint_and_candidate454() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    weights = _load_weights_module()
    model_bytes = zlib.decompress(base64.b85decode(weights.MODEL_B85_ZLIB))
    candidate_bytes = zlib.decompress(
        base64.b85decode(weights.CANDIDATE_SPEC_B85_ZLIB)
    )
    candidate_spec = json.loads(candidate_bytes)

    assert model_bytes == CHECKPOINT.read_bytes()
    assert hashlib.sha256(model_bytes).hexdigest() == manifest["checkpoint_sha256"]
    assert hashlib.sha256(model_bytes).hexdigest() == weights.MODEL_SHA256
    assert hashlib.sha256(candidate_bytes).hexdigest() == weights.CANDIDATE_SPEC_SHA256
    assert len(candidate_spec["candidate_ids"]) == manifest["candidate_count"] == 454
    assert len(set(candidate_spec["candidate_ids"])) == 454


def test_x_tree_notebook_is_thin_inference_entrypoint() -> None:
    notebook = json.loads(
        (PACKAGE / "unified_x_tree.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert code.strip() == "from unified_x_tree import main"
    language = notebook["metadata"]["language_info"]
    assert language["file_extension"] == ".py"
    assert language["nbconvert_exporter"] == "python"


def test_bundled_read_columns_helper_is_defined(tmp_path: Path) -> None:
    components = _load_components_module()
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"date": ["2024-01-02"], "value": [1.0]}).to_parquet(
        path,
        index=False,
    )
    loaded = components.read_columns(path, ["value"])
    assert loaded.to_dict(orient="list") == {"value": [1.0]}
