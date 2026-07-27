"""Build the frozen joint Elastic Net submission notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "factor_joint_elastic_net.py"
TARGET = ROOT / "submissions" / "factor_joint_elastic_net.ipynb"


def rendered_notebook() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    "# BigAlpha 2026 joint Elastic Net submission\n",
                    "\n",
                    (
                        "冻结的 15 个公开因子与 11 个通过增量门槛的自研因子联合模型。"
                        "测试期只重建 PIT 特征并应用测试期前冻结的模型权重。"
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = rendered_notebook()
    if args.check:
        return int(not TARGET.exists() or TARGET.read_text(encoding="utf-8") != rendered)
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
