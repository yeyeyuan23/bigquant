"""Build a submission from the latest orthogonal T result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.audit_submission_candidate_eligibility import (
        filter_candidate_ids,
    )
except ModuleNotFoundError:
    from audit_submission_candidate_eligibility import filter_candidate_ids
from submission_builder_support import (
    submission_runtime_source,
    write_candidate_module,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_HELPERS_SOURCE = (
    ROOT / "scripts/assets/external_submission_helpers.py.txt"
)

FAST_GROUP_ROLLING_SOURCE = r'''

_pandas_group_rolling = _group_rolling


def _group_rolling(
    frame,
    values,
    *,
    window,
    statistic,
    center=False,
    min_periods=None,
):
    """Equivalent fixed-row grouped rolling without pandas MultiIndex overhead."""
    if center:
        return _pandas_group_rolling(
            frame,
            values,
            window=window,
            statistic=statistic,
            center=True,
            min_periods=min_periods,
        )
    group_start = (
        frame["instrument"].ne(frame["instrument"].shift())
        | frame["trade_date"].ne(frame["trade_date"].shift())
        | frame["session_id"].ne(frame["session_id"].shift())
    )
    group_codes = group_start.cumsum()
    numeric = pd.to_numeric(values, errors="coerce").reindex(frame.index)
    valid = numeric.notna().astype("int64")
    filled = numeric.fillna(0.0)
    minimum = window if min_periods is None else min_periods

    count_cumulative = valid.groupby(group_codes, sort=False).cumsum()
    sum_cumulative = filled.groupby(group_codes, sort=False).cumsum()
    count = count_cumulative - count_cumulative.groupby(
        group_codes,
        sort=False,
    ).shift(window, fill_value=0)
    total = sum_cumulative - sum_cumulative.groupby(
        group_codes,
        sort=False,
    ).shift(window, fill_value=0)

    if statistic == "sum":
        result = total
    elif statistic == "mean":
        result = total / count.where(count.gt(0))
    elif statistic == "std":
        square_cumulative = filled.pow(2).groupby(
            group_codes,
            sort=False,
        ).cumsum()
        total_square = square_cumulative - square_cumulative.groupby(
            group_codes,
            sort=False,
        ).shift(window, fill_value=0)
        variance = (
            total_square - total.pow(2) / count.where(count.gt(0))
        ) / (count - 1.0).where(count.gt(1))
        result = np.sqrt(variance.clip(lower=0.0))
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")

    result = result.where(count.ge(minimum))
    return result.reindex(frame.index)
'''


def external_helpers_source() -> str:
    source = EXTERNAL_HELPERS_SOURCE.read_text(encoding="utf-8")
    return "\n" + source.rstrip() + FAST_GROUP_ROLLING_SOURCE


def render_notebook(
    source: str,
    candidate_count: int,
    screened15_lambda: float,
) -> str:
    mode = "no15" if screened15_lambda == 0.0 else "add15"
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    (
                        "# BigAlpha 2026 orthogonal T LightGBM "
                        f"({candidate_count} factors, {mode})\n"
                    ),
                    (
                        "Frozen orthogonal T pool; screened15 residual target; "
                        f"screened15 output lambda={screened15_lambda:g}; "
                        "causal rolling 60-day training "
                        "and 20-day prediction blocks with a one-day label embargo."
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
    parser.add_argument(
        "--output-stem",
        type=Path,
        help=(
            "output path without .py/.ipynb; defaults to "
            "remote_submission_notebooks/lgbm_t_orthogonal_N_candidate"
        ),
    )
    parser.add_argument(
        "--screened15-lambda",
        type=float,
        default=1.0,
        help=(
            "weight added back from the screened15 baseline after fitting the "
            "same residual-target T model; must be between 0 and 1"
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.screened15_lambda <= 1.0:
        raise ValueError("--screened15-lambda must be between 0 and 1")

    result = json.loads(args.result.read_text(encoding="utf-8"))
    admitted = result.get("tree_admitted_candidates")
    if not isinstance(admitted, list) or not admitted:
        raise ValueError("T result contains no tree_admitted_candidates")
    raw_candidate_ids = [
        str(value).removeprefix("self__") for value in admitted
    ]
    if len(raw_candidate_ids) != len(set(raw_candidate_ids)):
        raise ValueError("T result contains duplicate admitted candidates")
    candidate_ids, excluded_candidates = filter_candidate_ids(
        raw_candidate_ids
    )

    output_stem = args.output_stem or (
        ROOT
        / "remote_submission_notebooks"
        / f"lgbm_t_orthogonal_{len(candidate_ids)}_candidate"
    )
    if not output_stem.is_absolute():
        output_stem = ROOT / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_py = output_stem.with_suffix(".py")
    output_nb = output_stem.with_suffix(".ipynb")
    dependency = output_stem.with_name(
        f"{output_stem.name}_deps"
    ).with_suffix(".py")
    write_candidate_module(candidate_ids, dependency)

    mode = "no15" if args.screened15_lambda == 0.0 else "add15"
    source = (
        (
            f'"""Orthogonal T LightGBM with {len(candidate_ids)} factors; '
            f'{mode}, screened15 lambda={args.screened15_lambda:g}."""\n\n'
        )
        + "# Auto-generated from the frozen orthogonal T artifact. Do not edit by hand.\n"
        + f"# Requires the generated sibling {dependency.name} module.\n"
        + external_helpers_source()
        + submission_runtime_source(
            candidate_ids,
            companion_module=dependency.stem,
            screened15_lambda=args.screened15_lambda,
            lean_market_runtime=True,
        )
    )
    output_py.write_text(source, encoding="utf-8")
    output_nb.write_text(
        render_notebook(
            source,
            len(candidate_ids),
            args.screened15_lambda,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_ids),
                "candidates": candidate_ids,
                "excluded_candidates": excluded_candidates,
                "screened15_lambda": args.screened15_lambda,
                "mode": mode,
                "source": str(output_py.relative_to(ROOT)),
                "notebook": str(output_nb.relative_to(ROOT)),
                "dependency": str(dependency.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
