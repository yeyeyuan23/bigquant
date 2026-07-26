"""Build the two frozen competition submission notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = (
    (
        ROOT / "submissions" / "factor_self_family_rank.py",
        ROOT / "submissions" / "factor_self_family_rank.ipynb",
        "# BigAlpha 2026 family-balanced rank composite\n",
        "2019—2021 准入后冻结的 FR/PV 家族等权规则复合。",
    ),
    (
        ROOT / "submissions" / "factor_joint_lightgbm.py",
        ROOT / "submissions" / "factor_joint_lightgbm.ipynb",
        "# BigAlpha 2026 rolling LightGBM\n",
        "冻结特征准入与超参数，按 60 个交易日训练、20 个交易日预测。",
    ),
)


def rendered_notebook(source_path: Path, title: str, description: str) -> str:
    source = source_path.read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [title, "\n", description],
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for source, target, title, description in SUBMISSIONS:
        rendered = rendered_notebook(source, title, description)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                stale.append(target)
            continue
        target.write_text(rendered, encoding="utf-8")
        print(f"WROTE: {target.relative_to(ROOT)}")
    return int(bool(stale))


if __name__ == "__main__":
    raise SystemExit(main())
