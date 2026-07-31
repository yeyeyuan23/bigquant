"""Build and validate the latest S, I and T submission notebooks."""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_reports_dir() -> Path:
    reports_root = ROOT / "reports"
    candidates = [reports_root, *reports_root.iterdir()]
    complete = [
        path
        for path in candidates
        if path.is_dir()
        and (path / "latest/s_only_result.json").is_file()
        and (path / "latest/i_only_result.json").is_file()
        and (path / "latest/t_orthogonal_only_result.json").is_file()
    ]
    if not complete:
        raise FileNotFoundError(
            "no reports directory contains both latest I and T results"
        )
    return max(
        complete,
        key=lambda path: max(
            (path / "latest/s_only_result.json").stat().st_mtime_ns,
            (path / "latest/i_only_result.json").stat().st_mtime_ns,
            (path / "latest/t_orthogonal_only_result.json").stat().st_mtime_ns,
        ),
    )


def run_builder(script: str, result: Path) -> tuple[Path, Path, Path, int]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--result",
            str(result),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    source = ROOT / payload["source"]
    notebook = ROOT / payload["notebook"]
    dependency = ROOT / payload["dependency"]
    return source, notebook, dependency, int(payload["candidate_count"])


def validate_pair(
    source: Path,
    notebook: Path,
    dependency: Path,
    candidate_count: int,
) -> None:
    py_compile.compile(str(source), doraise=True)
    source_text = source.read_text(encoding="utf-8")
    ast.parse(source_text)
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in payload["cells"] if cell["cell_type"] == "code"
    ]
    if len(code_cells) != 1:
        raise ValueError(f"{notebook} must contain exactly one code cell")
    if "".join(code_cells[0]["source"]) != source_text:
        raise ValueError(f"{notebook} code does not match {source}")
    if "_install_bigalpha_candidate_modules" in source_text or "exec(compile(" in source_text:
        raise ValueError(f"{source} still embeds candidate module source")
    if not dependency.is_file():
        raise ValueError(f"{source} is missing sibling dependency {dependency.name}")
    py_compile.compile(str(dependency), doraise=True)
    dependency_text = dependency.read_text(encoding="utf-8")
    dependency_tree = ast.parse(dependency_text)
    expected_import = f"from {dependency.stem} import get_candidate_spec"
    if expected_import not in source_text:
        raise ValueError(f"{source} does not import {dependency.name}")
    package_imports = [
        node
        for node in ast.walk(dependency_tree)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("bigalpha2026")
        )
        or (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("bigalpha2026") for alias in node.names)
        )
    ]
    if package_imports or "exec(" in dependency_text:
        raise ValueError(f"{dependency} is not a flat static dependency")
    if candidate_count < 1:
        raise ValueError(f"{source} contains no frozen candidates")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="explicit reports directory; otherwise use the newest complete I/T run",
    )
    args = parser.parse_args()
    reports_dir = args.reports_dir or discover_reports_dir()
    if args.reports_dir is not None and not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir

    builds = [
        run_builder(
            "build_s_family_submission.py",
            reports_dir / "latest/s_only_result.json",
        ),
        run_builder(
            "build_i_elasticnet_submission.py",
            reports_dir / "latest/i_only_result.json",
        ),
        run_builder(
            "build_t_orthogonal_submission.py",
            reports_dir / "latest/t_orthogonal_only_result.json",
        ),
    ]
    rows = []
    for source, notebook, dependency, candidate_count in builds:
        validate_pair(source, notebook, dependency, candidate_count)
        rows.append(
            {
                "source": str(source.relative_to(ROOT)),
                "notebook": str(notebook.relative_to(ROOT)),
                "dependency": str(dependency.relative_to(ROOT)),
                "candidate_count": candidate_count,
                "validation": "ok",
            }
        )
    print(
        json.dumps(
            {
                "reports_dir": str(reports_dir.relative_to(ROOT)),
                "builds": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
