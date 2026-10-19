"""Audit retained GTJA formula lookbacks against the half-year policy."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

SOURCE_PATTERN = re.compile(r"^GTJA-(\d{4})$")
FUNCTION_PATTERN = re.compile(r"^gtja_(\d{3})$")

# Positional arguments containing a historical window.  An integer means
# one position; a tuple means every listed position is a window.
WINDOW_ARGUMENTS: dict[str, int | tuple[int, ...]] = {
    "corr": 2,
    "covariance": 2,
    "decay_linear": 1,
    "delay": 1,
    "delta": 1,
    "highday": 1,
    "lowday": 1,
    "mean": 1,
    "regbeta": 2,
    "regresi": 2,
    "sma": 1,
    "std_": 1,
    "sum_": 1,
    "ts_argmax": 1,
    "ts_argmin": 1,
    "ts_max": 1,
    "ts_min": 1,
    "ts_rank": 1,
    "wma": 1,
    "rolling_mean": 0,
    "rolling_std": 0,
    "rolling_sum": 0,
}


def literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def retained_sources(candidate_root: Path) -> dict[int, str]:
    selected: dict[int, str] = {}
    for path in sorted((candidate_root / "pv").glob("pv_*.py")):
        values = literal_assignments(path)
        source_id = values.get("SOURCE_RESEARCH_ID")
        candidate_id = values.get("CANDIDATE_ID")
        if not isinstance(source_id, str) or not isinstance(candidate_id, str):
            continue
        match = SOURCE_PATTERN.fullmatch(source_id)
        if match is not None:
            selected[int(match.group(1))] = candidate_id
    return selected


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def literal_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def function_lookbacks(node: ast.FunctionDef) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = call_name(child)
        positions = WINDOW_ARGUMENTS.get(name)
        if positions is None:
            continue
        if isinstance(positions, int):
            positions = (positions,)
        for position in positions:
            if position >= len(child.args):
                continue
            window = literal_int(child.args[position])
            if window is not None:
                results.append(
                    {"operator": name, "window": window, "line": child.lineno}
                )
        for keyword in child.keywords:
            if keyword.arg in {"window", "window_size"}:
                window = literal_int(keyword.value)
                if window is not None:
                    results.append(
                        {"operator": name, "window": window, "line": child.lineno}
                    )
    return results


def audit(
    candidate_root: Path,
    external_root: Path,
    max_trading_days: int,
) -> dict[str, object]:
    selected = retained_sources(candidate_root)
    formulas: dict[int, dict[str, object]] = {}
    gtja_root = external_root / "aurumq_rl/factors/gtja191"
    for path in sorted(gtja_root.glob("batch_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            match = FUNCTION_PATTERN.fullmatch(node.name)
            if match is None:
                continue
            number = int(match.group(1))
            if number not in selected:
                continue
            lookbacks = function_lookbacks(node)
            formulas[number] = {
                "candidate_id": selected[number],
                "source_id": f"GTJA-{number:04d}",
                "implementation": str(path),
                "lookbacks": lookbacks,
                "max_lookback": max(
                    (int(row["window"]) for row in lookbacks), default=1
                ),
            }
    missing = sorted(set(selected) - set(formulas))
    violations = sorted(
        (
            row
            for row in formulas.values()
            if int(row["max_lookback"]) > max_trading_days
        ),
        key=lambda row: str(row["candidate_id"]),
    )
    return {
        "schema_version": "gtja-lookback-audit-v2",
        "candidate_count": len(selected),
        "implemented_count": len(formulas),
        "max_allowed_trading_days": max_trading_days,
        "observed_max_lookback": max(
            (int(row["max_lookback"]) for row in formulas.values()), default=0
        ),
        "missing_implementations": missing,
        "violations": violations,
        "passed": len(selected) == 149 and not missing and not violations,
        "factors": sorted(
            formulas.values(), key=lambda row: str(row["candidate_id"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--max-trading-days", type=int, default=126)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.candidate_root,
        args.external_root,
        args.max_trading_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "factors"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
