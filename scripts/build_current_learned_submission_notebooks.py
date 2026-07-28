"""Build the two current learned-model competition notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = (
    (
        ROOT / "submissions" / "enet_v01.py",
        ROOT / "submissions" / "enet_v01.ipynb",
        "# BigAlpha 2026 current rolling Elastic Net\n",
        (
            "screened15 + 旧 I 冻结池（PV-014、FR-002）；正系数 Elastic Net；"
            "20 个交易日预测块；训练标签排除预测块前一交易日。"
        ),
    ),
    (
        ROOT / "submissions" / "enet_v02.py",
        ROOT / "submissions" / "enet_v02.ipynb",
        "# BigAlpha 2026 refreshed rolling Elastic Net v02\n",
        (
            "screened15 + 当前 I 冻结池 "
            "PV-008/PV-019/FR-006/HF-004/PV-011；正系数 Elastic Net；"
            "只用 start_date 前已完整观察到的历史标签训练。"
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v01.py",
        ROOT / "submissions" / "lgbm_v01.ipynb",
        "# BigAlpha 2026 current rolling LightGBM\n",
        (
            "screened15 + T 冻结池；正单调 LightGBM；"
            "60 个交易日训练、20 个交易日预测。"
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v02.py",
        ROOT / "submissions" / "lgbm_v02.ipynb",
        "# BigAlpha 2026 platform-safe rolling LightGBM v02\n",
        (
            "screened15 + 11个冻结自研因子；修正 DAI 查询边界，"
            "所有数据查询均不超过 end_date。"
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v03.py",
        ROOT / "submissions" / "lgbm_v03.ipynb",
        "# BigAlpha 2026 refreshed rolling LightGBM v03\n",
        (
            "screened15 + 昨日手工重跑确认的 T 池 "
            "PV-008/OB-003/PV-020/PV-013；沿用 v02 的平台安全查询边界；"
            "只用 start_date 前已完整观察到的历史标签训练。"
        ),
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
    stale: list[Path] = []
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
