#!/usr/bin/env python3
"""Build deterministic, self-contained BigQuant submission notebooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "src" / "bigalpha2026" / "common.py"
TARGETS = {
    ROOT / "submissions" / "factor_hf_pressure.ipynb": (
        ROOT / "src" / "bigalpha2026" / "hf_pressure.py",
        "持续盘口压力与价格反应不足",
    ),
    ROOT / "submissions" / "factor_quality_interaction.ipynb": (
        ROOT / "src" / "bigalpha2026" / "quality_interaction.py",
        "现金流质量与盘口确认",
    ),
}
PLATFORM_TARGETS = {
    ROOT / "submissions" / "factor_hf_pressure_v2.ipynb": (
        ROOT / "src" / "bigalpha2026" / "platform_hf_v2.py",
        "持续盘口压力与价格反应不足（官方模板兼容版）",
    ),
    ROOT / "submissions" / "factor_quality_interaction_v2.ipynb": (
        ROOT / "src" / "bigalpha2026" / "platform_quality_v2.py",
        "现金流质量与盘口确认（官方模板兼容版）",
    ),
}
RESEARCH_TARGET = ROOT / "research" / "constrained_search.ipynb"
RESEARCH_SOURCES = [
    ROOT / "src" / "bigalpha2026" / "common.py",
    ROOT / "src" / "bigalpha2026" / "hf_pressure.py",
    ROOT / "src" / "bigalpha2026" / "quality_interaction.py",
    ROOT / "src" / "bigalpha2026" / "evaluation.py",
    ROOT / "src" / "bigalpha2026" / "search.py",
    ROOT / "src" / "bigalpha2026" / "research.py",
]


def _strip_package_imports(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    skipping_common = False
    for line in lines:
        if line.startswith("from __future__ import"):
            continue
        if line.startswith("from .") and " import (" in line:
            skipping_common = True
            continue
        if line.startswith("from ."):
            continue
        if skipping_common:
            if line.strip() == ")":
                skipping_common = False
            continue
        output.append(line)
    return "\n".join(output).strip() + "\n"


def render_notebook(factor_path: Path, title: str) -> str:
    common_source = _strip_package_imports(COMMON.read_text(encoding="utf-8"))
    factor_source = _strip_package_imports(factor_path.read_text(encoding="utf-8"))
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3.11.8",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# BigQuant single-factor submission\n\n"
            f"因子：**{title}**。这是一次独立提交，只返回 `date, instrument, factor`。",
            id="factor-description",
        ),
        nbformat.v4.new_code_cell(
            common_source + "\n\n" + factor_source,
            id="factor-code",
        ),
    ]
    return nbformat.writes(notebook, version=4)


def render_platform_notebook(factor_path: Path, title: str) -> str:
    """Render a minimal notebook that mirrors the official competition template."""

    factor_source = _strip_package_imports(factor_path.read_text(encoding="utf-8"))
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3.11.8",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# BigAlpha 2026 single-factor submission\n\n"
            f"因子：**{title}**。入口和数据源切换方式与官方模板一致。",
            id="factor-description",
        ),
        nbformat.v4.new_code_cell(factor_source, id="factor-code"),
    ]
    return nbformat.writes(notebook, version=4)


def render_research_notebook() -> str:
    source = "\n\n".join(
        _strip_package_imports(path.read_text(encoding="utf-8"))
        for path in RESEARCH_SOURCES
    )
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3.11.8",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# BigAlpha 2026 约束搜索\n\n"
            "在AIStudio内生成48个候选。先用小样本核对字段，再切换4C/16G运行完整历史。",
            id="research-description",
        ),
        nbformat.v4.new_code_cell(source, id="research-library"),
        nbformat.v4.new_markdown_cell(
            "## 运行入口\n\n"
            "下面代码默认注释，确认字段和资源后逐段运行。结果只保存在平台内。",
            id="research-instructions",
        ),
        nbformat.v4.new_code_cell(
            "# datasources = {}\n"
            "# specs = enumerate_candidate_specs()\n"
            "# candidate_library = build_candidate_library(\n"
            "#     datasources, '2019-01-01', '2024-12-31', specs\n"
            "# )\n"
            "# daily_features = load_daily_hf_features(\n"
            "#     datasources, '2019-01-01', '2024-12-31'\n"
            "# )\n"
            "# daily_prices = daily_prices_from_hf_features(daily_features)\n"
            "# result = evaluate_candidate_library(\n"
            "#     candidate_library, daily_prices,\n"
            "#     exposures=None, factorlib=None, specs=specs\n"
            "# )\n"
            "# result.summary.head(20)",
            id="research-entrypoint",
        ),
    ]
    return nbformat.writes(notebook, version=4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated notebooks do not match the source files.",
    )
    args = parser.parse_args()
    failed = False
    for target, (source, title) in TARGETS.items():
        rendered = render_notebook(source, title)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                print(f"OUTDATED: {target.relative_to(ROOT)}")
                failed = True
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            print(f"WROTE: {target.relative_to(ROOT)}")
    for target, (source, title) in PLATFORM_TARGETS.items():
        rendered = render_platform_notebook(source, title)
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                print(f"OUTDATED: {target.relative_to(ROOT)}")
                failed = True
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            print(f"WROTE: {target.relative_to(ROOT)}")
    research_rendered = render_research_notebook()
    if args.check:
        if (
            not RESEARCH_TARGET.exists()
            or RESEARCH_TARGET.read_text(encoding="utf-8") != research_rendered
        ):
            print(f"OUTDATED: {RESEARCH_TARGET.relative_to(ROOT)}")
            failed = True
    else:
        RESEARCH_TARGET.parent.mkdir(parents=True, exist_ok=True)
        RESEARCH_TARGET.write_text(research_rendered, encoding="utf-8")
        print(f"WROTE: {RESEARCH_TARGET.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
