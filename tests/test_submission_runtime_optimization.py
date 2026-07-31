from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.submission_builder_support import submission_runtime_source

ROOT = Path(__file__).resolve().parents[1]
I55 = ROOT / "remote_submission_notebooks" / "enet_i_55_candidate.py"
T28_VARIANTS = (
    ROOT / "submissions" / "lgbm_t_orthogonal_28_no15_candidate.py",
    ROOT / "submissions" / "lgbm_t_orthogonal_28_add15_candidate.py",
)
def test_generated_learned_submissions_keep_fast_runtime_contract():
    for path in (I55, *T28_VARIANTS):
        source = path.read_text(encoding="utf-8")
        assert "_LEAN_MARKET_RUNTIME = True" in source
        assert "chunk_days = 31" in source
        assert "component_specs = {}" in source
        assert "np.searchsorted(" in source
        assert "_pandas_group_rolling = _group_rolling" in source


def test_generated_runtime_keeps_long_history_and_complete_output_contract():
    source = submission_runtime_source(
        ["PV-009"],
        lean_market_runtime=True,
    )

    assert "bar5m_start = history_start" in source
    assert "requested_prediction_dates" in source
    assert 'result["factor"]' in source
    assert ".fillna(0.0)" in source
    assert "no prediction dates have a complete causal" not in source


def test_s_runtime_passes_numeric_libraries_to_candidate_builder():
    source = (
        ROOT / "remote_submission_notebooks" / "rule_s_59_candidate.py"
    ).read_text(encoding="utf-8")
    candidate_call = source.split(
        "factors = _candidate_factors(",
        maxsplit=1,
    )[1].split(")", maxsplit=1)[0]

    assert "pd," in candidate_call
    assert "np," in candidate_call


def test_fast_group_rolling_matches_embedded_pandas_reference():
    tree = ast.parse(I55.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_group_rolling"
    ]
    assert len(definitions) == 2
    functions = []
    for definition in definitions:
        namespace = {
            "np": np,
            "pd": pd,
            "_pandas_group_rolling": functions[0] if functions else None,
        }
        exec(  # noqa: S102 - execute only the two parsed local function nodes
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                str(I55),
                "exec",
            ),
            namespace,
        )
        functions.append(namespace["_group_rolling"])
    original, optimized = functions

    rng = np.random.default_rng(20260731)
    frame = pd.DataFrame(
        {
            "instrument": np.repeat(["A", "B", "C"], 48),
            "trade_date": pd.Timestamp("2023-01-03"),
            "session_id": np.tile(np.repeat(["AM", "PM"], 24), 3),
            "value": rng.normal(size=144),
        }
    )
    frame.loc[::11, "value"] = np.nan
    for statistic in ("sum", "mean", "std"):
        expected = original(
            frame,
            frame["value"],
            window=5,
            statistic=statistic,
            min_periods=3,
        )
        actual = optimized(
            frame,
            frame["value"],
            window=5,
            statistic=statistic,
            min_periods=3,
        )
        np.testing.assert_allclose(
            actual.to_numpy(),
            expected.to_numpy(),
            rtol=1e-11,
            atol=1e-11,
            equal_nan=True,
        )
