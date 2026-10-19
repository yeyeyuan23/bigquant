from __future__ import annotations

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED_PACKAGE = "bigalpha" + "2026"
TOP_LEVEL_MODULES = (
    "candidate_transforms",
    "competition_score_proxy",
    "evaluation",
    "factor_pool",
    "factorlib",
    "feature_contracts",
    "research_policy",
)


def test_src_tree_has_no_removed_wrapper() -> None:
    assert not (ROOT / "src" / REMOVED_PACKAGE).exists()


def test_flattened_top_level_modules_are_importable() -> None:
    for module_name in TOP_LEVEL_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_flattened_packages_are_importable() -> None:
    assert importlib.import_module("alpha_models").__name__ == "alpha_models"
    assert importlib.import_module("candidates").__name__ == "candidates"


def test_source_and_runtime_code_do_not_reference_removed_package() -> None:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "experiments")
    stale: list[str] = []
    for base in roots:
        for path in base.rglob("*"):
            if path.suffix not in {".py", ".sh", ".md", ".toml", ".csv"}:
                continue
            if REMOVED_PACKAGE in path.read_text(encoding="utf-8"):
                stale.append(str(path.relative_to(ROOT)))
    assert stale == []
