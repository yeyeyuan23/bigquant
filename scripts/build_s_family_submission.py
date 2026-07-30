"""Build a self-contained family-balanced S submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_t_orthogonal_submission import external_helpers_source
from submission_builder_support import (
    discover_candidate_modules,
    installer_source,
    submission_runtime_source,
)

ROOT = Path(__file__).resolve().parents[1]


def family_runtime_source(candidate_ids: list[str]) -> str:
    runtime = submission_runtime_source(candidate_ids)
    helpers, marker, _model_main = runtime.partition(
        "\ndef main(datasources, start_date, end_date):"
    )
    if not marker:
        raise RuntimeError("submission runtime has no replaceable main()")
    return (
        helpers
        + f'''

def main(datasources, start_date, end_date):
    import numpy as np
    import pandas as pd

    candidate_ids = {candidate_ids!r}
    start_ts, end_ts, _history_start, _public_columns, pool, factorlib, exposure, financial, daily_features = _load_common_inputs(
        datasources, start_date, end_date, pd, np
    )
    factors = _candidate_factors(
        candidate_ids,
        financial,
        factorlib,
        exposure,
        daily_features,
        pool,
    )
    long_parts = []
    for candidate_id, frame in factors.items():
        part = frame[["date", "instrument", "factor"]].copy()
        part["candidate_id"] = candidate_id
        long_parts.append(part)
    candidate_wide = (
        pd.concat(long_parts, ignore_index=True)
        .pivot(
            index=["date", "instrument"],
            columns="candidate_id",
            values="factor",
        )
        .reset_index()
    )
    panel = pool.merge(
        candidate_wide,
        on=["date", "instrument"],
        how="left",
        validate="one_to_one",
    )
    for candidate_id in candidate_ids:
        panel[candidate_id] = _rank_center(
            pd.to_numeric(panel[candidate_id], errors="coerce"),
            panel["date"],
            np,
        )
    families = {{}}
    for candidate_id in candidate_ids:
        family = candidate_id.split("-", 1)[0]
        families.setdefault(family, []).append(candidate_id)
    if not families:
        raise ValueError("S submission has no factor families")
    family_columns = []
    for family, columns in sorted(families.items()):
        family_column = f"_family_{{family}}"
        panel[family_column] = panel[columns].mean(axis=1)
        family_columns.append(family_column)
    panel["factor"] = _rank_center(
        panel[family_columns].mean(axis=1),
        panel["date"],
        np,
    )
    result = (
        panel.loc[
            panel["date"].between(start_ts, end_ts),
            ["date", "instrument", "factor"],
        ]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    if (
        result.empty
        or result.duplicated(["date", "instrument"]).any()
        or result["factor"].isna().any()
        or not np.isfinite(result["factor"]).all()
    ):
        raise ValueError("invalid factor output")
    return result
'''
    )


def render_notebook(source: str, candidate_count: int) -> str:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    f"# BigAlpha 2026 family-balanced S ({candidate_count} factors)\n",
                    (
                        "Frozen S pool; daily cross-sectional ranks are averaged "
                        "within each data family and then equally across families."
                    ),
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "factor-code",
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11.8",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    admitted = result.get("admitted_candidates")
    if not isinstance(admitted, list) or not admitted:
        raise ValueError("S result contains no admitted_candidates")
    candidate_ids = [str(value).removeprefix("self__") for value in admitted]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("S result contains duplicate admitted candidates")

    output_stem = args.output_stem or (
        ROOT
        / "remote_submission_notebooks"
        / f"rule_s_{len(candidate_ids)}_candidate"
    )
    if not output_stem.is_absolute():
        output_stem = ROOT / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_py = output_stem.with_suffix(".py")
    output_nb = output_stem.with_suffix(".ipynb")
    source = (
        f'"""S-route family-balanced composite with {len(candidate_ids)} factors."""\n\n'
        "# Auto-generated from the frozen S artifact. Do not edit by hand.\n"
        + installer_source(discover_candidate_modules(candidate_ids))
        + external_helpers_source()
        + family_runtime_source(candidate_ids)
    )
    output_py.write_text(source, encoding="utf-8")
    output_nb.write_text(render_notebook(source, len(candidate_ids)), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_ids),
                "candidates": candidate_ids,
                "source": str(output_py.relative_to(ROOT)),
                "notebook": str(output_nb.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
