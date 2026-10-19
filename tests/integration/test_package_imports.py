from __future__ import annotations

import importlib

TOP_LEVEL_MODULES = (
    "candidate_transforms",
    "competition_score_proxy",
    "evaluation",
    "factor_pool",
    "factorlib",
    "feature_contracts",
    "research_policy",
)


def test_declared_top_level_modules_are_importable() -> None:
    for module_name in TOP_LEVEL_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_model_and_candidate_packages_are_importable() -> None:
    assert importlib.import_module("alpha_models").__name__ == "alpha_models"
    assert importlib.import_module("candidates").__name__ == "candidates"
