"""Build the frozen INT-001 notebook from its self-contained source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "smoke_v01.py"
TARGET = ROOT / "submissions" / "smoke_v01.ipynb"


def rendered_notebook() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factor-description",
                "metadata": {},
                "source": [
                    "# BigAlpha 2026 single-factor submission\n",
                    "\n",
                    (
                        "因子：**INT-001 FR-002/HF-001 等权秩组合**。"
                        "冻结版本 `combination_v2_2026-07-26`。"
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
