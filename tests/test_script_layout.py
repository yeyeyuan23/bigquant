from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAMES = (
    "run_candidate454_assembly_after_components.sh",
    "run_factor_wiki_2019_2024.sh",
    "run_unified_after_candidate454.sh",
    "run_unified_alpha_fusion_suite.sh",
    "run_unified_full_experts.sh",
)


def test_unified_runners_live_under_scripts() -> None:
    assert not list(ROOT.glob("run_*.sh"))
    for name in RUNNER_NAMES:
        assert (ROOT / "scripts/runners" / name).is_file(), name


def test_script_category_directories_are_populated() -> None:
    assert len(list((ROOT / "scripts/aistudio").glob("aistudio_*.py"))) == 4
    assert len(list((ROOT / "scripts/transfer").glob("*_transfer_release.sh"))) == 2


def test_relocated_shell_entries_have_valid_syntax() -> None:
    entries = [*(ROOT / "scripts/runners").glob("*.sh")]
    entries.extend((ROOT / "scripts/transfer").glob("*.sh"))
    for entry in entries:
        subprocess.run(["bash", "-n", entry], check=True)
