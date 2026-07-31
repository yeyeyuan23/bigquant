from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_submission_candidate_eligibility.py"


def load_module():
    spec = importlib.util.spec_from_file_location("submission_eligibility", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_active_candidate_tree_is_submission_eligible():
    module = load_module()
    rows = module.scan_candidates()
    assert len(rows) == 140
    assert all(row["eligible"] for row in rows)


def test_retired_candidates_are_filtered_from_old_cached_pools():
    module = load_module()
    kept, excluded = module.filter_candidate_ids(
        ["HF-001", "FR-005", "PV-009"]
    )
    assert kept == ["HF-001"]
    assert set(excluded) == {"FR-005", "PV-009"}


def test_report_persists_scanner_and_candidate_hashes(tmp_path):
    module = load_module()
    report = module.write_eligibility_report(
        tmp_path / "eligibility.json",
        tmp_path / "eligibility.csv",
    )
    assert report["scanner_sha256"]
    assert all(row["source_sha256"] for row in report["candidates"])
    assert (tmp_path / "eligibility.json").is_file()
    assert (tmp_path / "eligibility.csv").is_file()
