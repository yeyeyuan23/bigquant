from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_unified_m_l5_submission import build_deep_book_sql, build_notebook


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
