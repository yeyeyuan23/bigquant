from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY_ROOT = ROOT / "reports/dependencies/frozen_full_history_replay_2019_2024"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_every_six_year_replay_keeps_its_frozen_checkpoint() -> None:
    manifests = sorted(REPLAY_ROOT.glob("*/*_replay_manifest.json"))
    assert len(manifests) == 6

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest["checkpoint_artifact"]
        checkpoint = ROOT / artifact["path"]
        assert checkpoint.is_file(), manifest_path
        assert _sha256(checkpoint) == artifact["sha256"], manifest_path
        assert manifest["training_performed"] is False
        assert manifest["includes_checkpoint_training_period"] is True


def test_submission_checkpoint_manifest_references_are_local() -> None:
    bundle = json.loads(
        (ROOT / "submissions/m_l5/submission_bundle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (ROOT / bundle["checkpoint_manifest"]).is_file()


def test_selected_route_training_sources_are_retained() -> None:
    required_sources = (
        "scripts/evaluate_unified_tree.py",
        "scripts/train_unified_final_checkpoint.py",
        "scripts/evaluate_unified_temporal.py",
        "scripts/evaluate_unified_temporal_residual.py",
        "scripts/run_m_expanding_history_retrain.sh",
        "scripts/train_unified_microstructure_v3_final.py",
    )
    for relative_path in required_sources:
        assert (ROOT / relative_path).is_file(), relative_path

    required_provenance = (
        (
            "reports/dependencies/"
            "unified_alpha_fusion_suite_20260801_full_experts_v1/"
            "lightgbm/run_manifest.json"
        ),
        (
            "reports/dependencies/xt_protocol_retrain_20260803/"
            "t_temporal/screen_2023/d128_l2_dense_s1_e8/"
            "seed_20260803/command.json"
        ),
        (
            "reports/dependencies/xt_protocol_retrain_20260803/"
            "x_mlp/final_full_history/final_checkpoint.json"
        ),
        (
            "reports/dependencies/m_expanding_history_retrain_20260803/"
            "final_full_history_e3/run_manifest.json"
        ),
        (
            "reports/dependencies/m_v3_final_2019_2024_seed_20260803/"
            "final_checkpoint_manifest.json"
        ),
    )
    for relative_path in required_provenance:
        assert (ROOT / relative_path).is_file(), relative_path
