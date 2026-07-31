"""Generate the factor-track submission eligibility manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "src/bigalpha2026/candidates"
FORBIDDEN_INPUTS = {"exposure", "exposures", "factorlib"}
MAX_DAILY_OBSERVATIONS = 126
MAX_CALENDAR_MONTHS = 6
ID_RE = re.compile(r"^(fr|hf|ob|pv|int)_(\d{3})$")
RETIRED_CANDIDATES = {
    "FR-005": ["forbidden_input:exposures"],
    "FR-008": ["calendar_history:36>6m"],
    "FR-009": ["calendar_history:60>6m"],
    "FR-011": ["forbidden_input:exposures"],
    "FR-012": ["calendar_history:12>6m"],
    "FR-014": ["forbidden_input:exposures"],
    "PV-008": ["daily_history:252>126"],
    "PV-009": ["daily_history:252>126"],
    "PV-010": ["daily_history:252>126"],
    "PV-011": ["daily_history:252>126"],
    "PV-012": ["forbidden_input:exposures"],
    "PV-013": ["daily_history:252>126"],
    "PV-015": ["calendar_history:36>6m"],
    "PV-016": ["forbidden_input:factorlib", "calendar_history:36>6m"],
    "PV-017": ["calendar_history:60>6m"],
    "PV-018": ["daily_history:756>126"],
    "PV-019": ["daily_history:252>126"],
    "PV-022": ["daily_history:252>126"],
    "INT-003": ["transitive_dependency:FR-012"],
}


def number(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    return None


def defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    positional = [*node.args.posonlyargs, *node.args.args]
    default_count = len(node.args.defaults)
    names = (
        []
        if default_count == 0
        else [item.arg for item in positional[-default_count:]]
    )
    pairs = list(zip(names, node.args.defaults, strict=True))
    pairs.extend(zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True))
    return {
        (name if isinstance(name, str) else name.arg): value
        for name, default in pairs
        if (value := number(default)) is not None
    }


def required_inputs(tree: ast.Module) -> list[str]:
    result = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("build_") or "_factor" not in node.name:
            continue
        positional = [*node.args.posonlyargs, *node.args.args]
        required_count = len(positional) - len(node.args.defaults)
        result.update(item.arg for item in positional[:required_count])
    return sorted(result)


def candidate_dependencies(tree: ast.Module) -> list[str]:
    dependencies = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        stem = node.module.rsplit(".", 1)[-1]
        match = ID_RE.match(stem)
        if match is not None:
            dependencies.add(
                f"{match.group(1).upper()}-{match.group(2)}"
            )
    return sorted(dependencies)


def history(tree: ast.Module, family: str) -> tuple[int, int, list[str]]:
    daily = 0
    months = 0
    evidence = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        node_defaults = defaults(node)
        for name, value in node_defaults.items():
            if value <= 0:
                continue
            if name == "years":
                months = max(months, value * 12)
                evidence.append(f"{node.name}:{name}={value}y")
            elif "monthly" in node.name and name in {
                "window", "min_periods", "history_window",
            }:
                months = max(months, value)
                evidence.append(f"{node.name}:{name}={value}m")
            elif (
                family == "FR"
                and (
                    name == "min_periods"
                    or (
                        name == "history_window"
                        and "min_periods" not in node_defaults
                    )
                )
            ):
                months = max(months, value * 3)
                evidence.append(f"{node.name}:{name}={value} disclosures")
            elif (
                "lag" in name
                or "window" in name
                or "lookback" in name
                or name in {"min_periods", "formation_days", "skip_days"}
            ):
                daily = max(daily, value)
                evidence.append(f"{node.name}:{name}={value}d")
    return daily, months, evidence


def audit(path: Path) -> dict[str, object] | None:
    match = ID_RE.match(path.stem)
    if match is None:
        return None
    factor_id = f"{match.group(1).upper()}-{match.group(2)}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    inputs = required_inputs(tree)
    dependencies = candidate_dependencies(tree)
    forbidden = sorted(set(inputs) & FORBIDDEN_INPUTS)
    daily, months, evidence = history(tree, factor_id.split("-")[0])
    reasons = []
    if forbidden:
        reasons.append("forbidden_input:" + ",".join(forbidden))
    if daily > MAX_DAILY_OBSERVATIONS:
        reasons.append(f"daily_history:{daily}>{MAX_DAILY_OBSERVATIONS}")
    if months > MAX_CALENDAR_MONTHS:
        reasons.append(f"calendar_history:{months}>{MAX_CALENDAR_MONTHS}m")
    return {
        "candidate_id": factor_id,
        "eligible": not reasons,
        "required_inputs": inputs,
        "candidate_dependencies": dependencies,
        "forbidden_inputs": forbidden,
        "required_daily_observations": daily,
        "required_calendar_months": months,
        "reasons": reasons,
        "history_evidence": evidence,
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
    }


def scan_candidates() -> list[dict[str, object]]:
    rows = [
        row
        for path in sorted(CANDIDATES.glob("*/*.py"))
        if (row := audit(path)) is not None
    ]
    rows.sort(key=lambda row: str(row["candidate_id"]))
    by_id = {str(row["candidate_id"]): row for row in rows}
    changed = True
    while changed:
        changed = False
        for row in rows:
            transitive = [
                dependency
                for dependency in row["candidate_dependencies"]
                if (
                    dependency in RETIRED_CANDIDATES
                    or (
                        dependency in by_id
                        and not by_id[dependency]["eligible"]
                    )
                )
            ]
            reasons = list(row["reasons"])
            additions = [
                f"transitive_dependency:{dependency}"
                for dependency in transitive
                if f"transitive_dependency:{dependency}" not in reasons
            ]
            if additions:
                row["reasons"] = [*reasons, *additions]
                row["eligible"] = False
                changed = True
    return rows


def filter_candidate_ids(
    candidate_ids: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    decisions = {
        str(row["candidate_id"]): row
        for row in scan_candidates()
    }
    missing = sorted(set(candidate_ids) - set(decisions))
    retired = sorted(set(missing) & set(RETIRED_CANDIDATES))
    missing = sorted(set(missing) - set(retired))
    if missing:
        raise ValueError(f"candidate eligibility is unknown: {missing}")
    excluded = {
        candidate_id: list(decisions[candidate_id]["reasons"])
        for candidate_id in candidate_ids
        if (
            candidate_id in decisions
            and not decisions[candidate_id]["eligible"]
        )
    }
    excluded.update(
        {
            candidate_id: RETIRED_CANDIDATES[candidate_id]
            for candidate_id in retired
        }
    )
    kept = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in excluded
    ]
    if not kept:
        raise ValueError("submission eligibility filter removed every candidate")
    return kept, excluded


def candidate_pool_availability(
    candidate_ids: list[str],
    *,
    required_candidate_ids: list[str] | None = None,
) -> dict[str, object]:
    rows = scan_candidates()
    active_eligible = {
        str(row["candidate_id"])
        for row in rows
        if bool(row["eligible"])
    }
    available = set(candidate_ids)
    required = (
        active_eligible
        if required_candidate_ids is None
        else set(required_candidate_ids)
    )
    return {
        "schema_version": "candidate-pool-availability-v1",
        "active_eligible_count": len(active_eligible),
        "available_candidate_count": len(available),
        "required_candidate_count": len(required),
        "missing_required_count": len(required - available),
        "missing_required_candidates": sorted(required - available),
        "available_without_active_source": sorted(
            available - active_eligible - set(RETIRED_CANDIDATES)
        ),
        "available_retired_candidates": sorted(
            available & set(RETIRED_CANDIDATES)
        ),
    }


def write_candidate_pool_availability_report(
    path: Path,
    candidate_ids: list[str],
    *,
    required_candidate_ids: list[str] | None = None,
) -> dict[str, object]:
    report = candidate_pool_availability(
        candidate_ids,
        required_candidate_ids=required_candidate_ids,
    )
    report["scanner_source"] = str(Path(__file__).resolve().relative_to(ROOT))
    report["scanner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def write_eligibility_report(
    json_path: Path,
    csv_path: Path,
) -> dict[str, object]:
    rows = scan_candidates()
    report = {
        "schema_version": "factor-track-submission-eligibility-v1",
        "policy": {
            "factor_inputs": ["bar1m", "financial"],
            "auxiliary_inputs": ["instruments"],
            "forbidden_inputs": sorted(FORBIDDEN_INPUTS),
            "maximum_history_months": MAX_CALENDAR_MONTHS,
            "maximum_daily_observations": MAX_DAILY_OBSERVATIONS,
        },
        "candidate_count": len(rows),
        "eligible_count": sum(bool(row["eligible"]) for row in rows),
        "ineligible_count": sum(not bool(row["eligible"]) for row in rows),
        "retired_count": len(RETIRED_CANDIDATES),
        "retired_candidates": RETIRED_CANDIDATES,
        "candidates": rows,
    }
    report["scanner_source"] = str(Path(__file__).resolve().relative_to(ROOT))
    report["scanner_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **{
                        key: "|".join(row[key])
                        for key in (
                            "required_inputs",
                            "candidate_dependencies",
                            "forbidden_inputs",
                            "reasons",
                            "history_evidence",
                        )
                    },
                }
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "reports/submission_candidate_eligibility.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "reports/submission_candidate_eligibility.csv",
    )
    args = parser.parse_args()
    report = write_eligibility_report(args.json, args.csv)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("candidate_count", "eligible_count", "ineligible_count")
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
