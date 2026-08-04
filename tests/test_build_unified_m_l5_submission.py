from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_unified_m_l5_submission import (
    build_deep_book_sql,
    build_handoff,
    build_notebook,
)


def test_deep_book_sql_is_bounded_and_uses_all_five_levels() -> None:
    sql = build_deep_book_sql(
        "bigalpha_2026_stock_bar1m",
        "2025-01-02",
        "2025-01-09",
    )

    assert "bid_price5" in sql
    assert "ask_volume5" in sql
    assert "date >= TIMESTAMP '2025-01-02'" in sql
    assert "date < TIMESTAMP '2025-01-09'" in sql
    assert "negative_mid_shock_q10_bid_depth_recovery_5m_median" in sql


def test_submission_notebook_is_a_single_thin_entrypoint(tmp_path: Path) -> None:
    path = tmp_path / "unified_m_l5.ipynb"
    second_path = tmp_path / "unified_m_l5_second.ipynb"

    build_notebook(path)
    build_notebook(second_path)

    notebook = nbformat.read(path, as_version=4)
    assert len(notebook.cells) == 1
    assert notebook.cells[0].cell_type == "code"
    assert (
        notebook.cells[0].source
        == "from unified_m_l5 import main  # noqa: F401\n"
    )
    assert notebook.cells[0].id == "unified-m-l5-entrypoint"
    assert path.read_bytes() == second_path.read_bytes()


def test_handoff_supports_submission_subdirectory() -> None:
    handoff = build_handoff("M-l5 test handoff")
    lines = handoff.splitlines()

    assert 'M_L5_DIR="${M_L5_DIR:-/home/aiuser/work/sub_m_l5}"' in handoff
    assert 'PYTHONPATH="$M_L5_DIR" python "$M_L5_PROBE"' in handoff
    assert 'PYTHONPATH="$M_L5_DIR" python "$M_L5_PROBE" \\' in lines
    assert '  "$M_L5_DIR/unified_m_l5.py" \\' in lines
    assert '"$M_L5_DIR/unified_m_l5.py"' in handoff
    assert 'tee "$M_L5_DIR/unified_m_l5_probe.log"' in handoff
    assert "python -m jupyter nbconvert --to notebook --execute" in handoff
    assert "--output unified_m_l5_executed.ipynb" in handoff
    assert "unified_m_l5_notebook.py" not in handoff
    assert "cd /home/aiuser/work\n" not in handoff
    assert "/home/aiuser/work/unified_m_l5.py" not in handoff
