from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_shell_entrypoints_have_valid_syntax() -> None:
    entries = [*(ROOT / "scripts/runners").glob("*.sh")]
    entries.extend((ROOT / "scripts/transfer").glob("*.sh"))
    assert entries, "No maintained shell entrypoints were collected"
    for entry in entries:
        subprocess.run(["bash", "-n", entry], check=True)
