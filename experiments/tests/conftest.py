from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALS_PRE = REPO_ROOT / "experiments/finals_pre"
CONTRACT_PATH = REPO_ROOT / "experiments/contracts/o2c_experiments.json"

for entry in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


@pytest.fixture(scope="session")
def contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def load_experiment_module():
    loaded: dict[tuple[str, str], ModuleType] = {}

    def load(relative_path: str, module_name: str) -> ModuleType:
        key = (relative_path, module_name)
        if key in loaded:
            return loaded[key]
        path = FINALS_PRE / relative_path
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import experiment module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[key] = module
        return module

    return load
