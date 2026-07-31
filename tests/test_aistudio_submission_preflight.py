from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "remote_submission_notebooks"
    / "aistudio_submission_preflight.py"
)


def load_preflight():
    spec = importlib.util.spec_from_file_location("submission_preflight", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_undefined_global_scan_detects_missing_import():
    preflight = load_preflight()
    broken = """
import numpy as np

def rank(frame):
    return pd.to_numeric(frame).replace([np.inf, -np.inf], np.nan)
"""
    assert preflight._undefined_globals(broken, "broken") == ["pd"]
    fixed = broken.replace("import numpy as np", "import numpy as np\nimport pandas as pd")
    assert preflight._undefined_globals(fixed, "fixed") == []


def test_current_submission_sources_pass_local_preflight():
    sources = [
        ROOT / "remote_submission_notebooks" / "rule_s_56_candidate.py",
        ROOT / "remote_submission_notebooks" / "enet_i_53_candidate.py",
        ROOT / "submissions" / "lgbm_t_orthogonal_26_no15_candidate.py",
        ROOT / "submissions" / "lgbm_t_orthogonal_26_add15_candidate.py",
    ]
    for source in sources:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(source)],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["status"] == "ok"
        assert payload["static"]["module_mode"] == "flat_dependency"


def test_platform_schema_queries_use_a_bounded_date_filter(monkeypatch):
    preflight = load_preflight()
    calls = []

    def query(sql, **kwargs):
        calls.append((sql, kwargs))
        selected = re.search(r"SELECT (.+) FROM ", sql).group(1)
        columns = [column.strip() for column in selected.split(",")]
        return SimpleNamespace(df=lambda: pd.DataFrame(columns=columns))

    monkeypatch.setitem(sys.modules, "dai", SimpleNamespace(query=query))
    result = preflight.platform_schema_preflight(
        ROOT / "remote_submission_notebooks" / "enet_i_53_candidate.py",
        "bigalpha_2026_financial",
        "2023-01-04",
    )
    assert result["status"] == "ok"
    assert len(calls) == 5
    for _, kwargs in calls:
        assert kwargs["filters"] == {
            "date": ["2023-01-04", "2023-01-04"]
        }
