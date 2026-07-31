from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
I55 = ROOT / "remote_submission_notebooks" / "enet_i_55_candidate.py"
T28_VARIANTS = (
    ROOT / "submissions" / "lgbm_t_orthogonal_28_no15_candidate.py",
    ROOT / "submissions" / "lgbm_t_orthogonal_28_add15_candidate.py",
)


def load_i55():
    spec = importlib.util.spec_from_file_location("optimized_i55", I55)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_learned_submissions_keep_fast_runtime_contract():
    for path in (I55, *T28_VARIANTS):
        source = path.read_text(encoding="utf-8")
        assert "_LEAN_MARKET_RUNTIME = True" in source
        assert "chunk_days = 31" in source
        assert "component_specs = {}" in source
        assert "np.searchsorted(" in source
        assert "_pandas_group_rolling = _group_rolling" in source


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


def test_batched_daily_component_factors_match_original_builders():
    module = load_i55()
    module._install_bigalpha_candidate_modules()
    candidate_ids = next(
        value
        for value in module.main.__code__.co_consts
        if isinstance(value, tuple) and "FR-005" in value
    )
    selected = []
    specifications = {}
    for candidate_id in candidate_ids:
        family, number = candidate_id.split("-")
        package = {"FR": "fr", "HF": "hf", "PV": "pv"}[family]
        candidate_module = importlib.import_module(
            f"bigalpha2026.candidates.{package}."
            f"{family.lower()}_{number}"
        )
        stem = candidate_id.lower().replace("-", "_")
        builder = getattr(
            candidate_module,
            f"build_{stem}_factor_from_daily",
            None,
        )
        if (
            builder is not None
            and hasattr(candidate_module, "COMPONENT_COLUMN")
            and hasattr(candidate_module, "ORIENTATION")
        ):
            selected.append(candidate_id)
            specifications[candidate_id] = (
                builder,
                candidate_module.COMPONENT_COLUMN,
            )

    pool = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2023-01-03", "2023-01-04"]),
            ["A", "B", "C", "D"],
        ],
        names=["date", "instrument"],
    ).to_frame(index=False)
    rng = np.random.default_rng(20260731)
    daily_features = pool.copy()
    for column in sorted({value[1] for value in specifications.values()}):
        daily_features[column] = rng.normal(size=len(pool))
        daily_features.loc[1, column] = np.nan

    complex_results, batched = module._candidate_factors(
        selected,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        daily_features,
        pool,
        pd,
        np,
    )
    assert complex_results == {}
    assert len(selected) == 44
    available = {
        "daily_features": daily_features,
        "daily_bars": daily_features,
        "bars": daily_features,
        "pv": daily_features,
        "micro": daily_features,
        "micro_daily": daily_features,
        "pool": pool,
    }
    for candidate_id, (builder, _) in specifications.items():
        arguments = [
            available[parameter.name]
            for parameter in inspect.signature(builder).parameters.values()
            if parameter.default is inspect.Parameter.empty
        ]
        expected = builder(*arguments)
        np.testing.assert_allclose(
            batched[candidate_id].to_numpy(),
            expected["factor"].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
