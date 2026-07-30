"""Build a self-contained Elastic Net submission from the latest I result."""

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
LIGHTGBM_MODEL = (
    'model = LGBMRegressor(objective="regression", learning_rate=0.03, '
    "n_estimators=220, max_depth=3, num_leaves=7, min_child_samples=100, "
    "subsample=1.0, colsample_bytree=0.8, reg_lambda=1.0, "
    "random_state=20260730, n_jobs=1, deterministic=True, "
    "force_col_wise=True, verbosity=-1, "
    "monotone_constraints=[1] * len(feature_columns))"
)
ELASTIC_NET_MODEL = (
    "model = ElasticNet(alpha=0.001, l1_ratio=0.5, fit_intercept=True, "
    "max_iter=20000, random_state=0, positive=True, selection=\"cyclic\")"
)


def elastic_net_runtime_source(candidate_ids: list[str]) -> str:
    runtime = submission_runtime_source(candidate_ids)
    runtime = runtime.replace(
        "from lightgbm import LGBMRegressor",
        "from sklearn.linear_model import ElasticNet",
    )
    runtime = runtime.replace(LIGHTGBM_MODEL, ELASTIC_NET_MODEL)
    if "LGBMRegressor" in runtime or ELASTIC_NET_MODEL not in runtime:
        raise RuntimeError("failed to replace LightGBM runtime with Elastic Net")
    return runtime


def render_notebook(source: str, candidate_count: int) -> str:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    f"# BigAlpha 2026 I-pool Elastic Net ({candidate_count} factors)\n",
                    (
                        "Frozen I pool; positive Elastic Net; causal rolling "
                        "60-day training and 20-day prediction blocks with a "
                        "one-day label embargo."
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
            "submissions/enet_i_N_candidate"
        ),
    )
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    frozen = result.get("frozen_candidates")
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("I result contains no frozen_candidates")
    candidate_ids = [str(value).removeprefix("self__") for value in frozen]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("I result contains duplicate frozen candidates")

    output_stem = args.output_stem or (
        ROOT / "submissions" / f"enet_i_{len(candidate_ids)}_candidate"
    )
    if not output_stem.is_absolute():
        output_stem = ROOT / output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_py = output_stem.with_suffix(".py")
    output_nb = output_stem.with_suffix(".ipynb")

    source = (
        f'"""I-route rolling Elastic Net with {len(candidate_ids)} factors."""\n\n'
        "# Auto-generated from the frozen I artifact. Do not edit by hand.\n"
        + installer_source(discover_candidate_modules(candidate_ids))
        + external_helpers_source()
        + elastic_net_runtime_source(candidate_ids)
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
