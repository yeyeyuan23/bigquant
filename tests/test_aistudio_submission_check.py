from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "remote_submission_notebooks" / "aistudio_submission_check.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("submission_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_check_rejects_forbidden_table_and_long_lookback(tmp_path):
    checker = load_checker()
    source = tmp_path / "candidate.py"
    source.write_text(
        "import pandas as pd\n"
        "TABLE = 'bigalpha_2026_stock_bar5m'\n"
        "def main(datasources, start_date, end_date):\n"
        "    return pd.Timestamp(start_date) - pd.DateOffset(days=500)\n",
        encoding="utf-8",
    )
    result = checker.source_check(source, 6)
    assert result["status"] == "error"
    assert result["forbidden_tables"] == ["bigalpha_2026_stock_bar5m"]
    assert result["excessive_lookback"]


def test_output_check_detects_missing_whole_day():
    checker = load_checker()
    universe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-04", "2023-01-05"]),
            "instrument": ["A", "A"],
        }
    )
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-04"]),
            "instrument": ["A"],
            "factor": [1.0],
        }
    )
    result = checker.output_check(
        output,
        universe,
        pd.Timestamp("2023-01-04"),
        pd.Timestamp("2023-01-05"),
    )
    assert result["status"] == "invalid_output"
    assert result["missing_date_sample"] == ["2023-01-05"]


def test_undefined_globals_detects_missing_pandas_import():
    checker = load_checker()
    broken = (
        "import numpy as np\n"
        "def rank(frame):\n"
        "    return pd.to_numeric(frame).replace([np.inf, -np.inf], np.nan)\n"
    )
    assert checker.undefined_globals(broken, "broken") == ["pd"]
