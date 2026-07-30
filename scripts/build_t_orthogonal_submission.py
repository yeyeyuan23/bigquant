"""Build a self-contained submission from the latest orthogonal T result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from submission_builder_support import (
    discover_candidate_modules,
    installer_source,
    submission_runtime_source,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_HELPERS_SOURCE = (
    ROOT / "scripts/assets/external_submission_helpers.py.txt"
)


def external_helpers_source() -> str:
    source = EXTERNAL_HELPERS_SOURCE.read_text(encoding="utf-8")
    return "\n" + source.rstrip() + "\n"


def render_notebook(source: str, candidate_count: int) -> str:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    f"# BigAlpha 2026 orthogonal T LightGBM ({candidate_count} factors)\n",
                    (
                        "Frozen orthogonal T pool; screened15 residual target; "
                        "self-only LightGBM output; causal rolling 60-day training "
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
            "submissions/lgbm_t_orthogonal_N_candidate"
        ),
    )
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    admitted = result.get("tree_admitted_candidates")
    if not isinstance(admitted, list) or not admitted:
        raise ValueError("T result contains no tree_admitted_candidates")
    candidate_ids = [str(value).removeprefix("self__") for value in admitted]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("T result contains duplicate admitted candidates")

    output_stem = args.output_stem or (
        ROOT
        / "submissions"
        / f"lgbm_t_orthogonal_{len(candidate_ids)}_candidate"
    )
    if not output_stem.is_absolute():
        output_stem = ROOT / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_py = output_stem.with_suffix(".py")
    output_nb = output_stem.with_suffix(".ipynb")

    source = (
        f'"""Orthogonal T pure-increment LightGBM with {len(candidate_ids)} factors."""\n\n'
        "# Auto-generated from the frozen orthogonal T artifact. Do not edit by hand.\n"
        + installer_source(discover_candidate_modules(candidate_ids))
        + external_helpers_source()
        + submission_runtime_source(candidate_ids)
    )
    output_py.write_text(source, encoding="utf-8")
    output_nb.write_text(
        render_notebook(source, len(candidate_ids)),
        encoding="utf-8",
    )
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
