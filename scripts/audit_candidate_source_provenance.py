"""Trace every active candidate to repository input and generator evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_submission_candidate_eligibility import scan_candidates

CANDIDATES = ROOT / "src/bigalpha2026/candidates"
ALLOWED_DIRECT_INPUTS = {
    "daily_bars": "bar1m",
    "minute_bars": "bar1m",
    "financial": "financial",
    "trading_days": "instruments",
    "pool": "instruments",
}
GENERATOR_BY_SOURCE_PREFIX = {
    "CICC": (
        "scripts/build_cicc34_candidate_pool_delta.py|"
        "scripts/build_factor_wiki_latent_hf_cicc13_delta.py"
    ),
    "CJ": "scripts/factor_wiki_remaining/build_changjiang_components.py",
    "FZ": "scripts/build_fz76_candidate_pool_delta.py",
    "HAITONG": (
        "scripts/build_factor_wiki_latent_pv_delta.py|"
        "scripts/factor_wiki_remaining/build_haitong_components.py"
    ),
}
IN_REPOSITORY_DAILY_FEATURE_GENERATORS = {
    "HF-001": "src/bigalpha2026/candidates/hf/hf_001.py",
    "HF-002": "src/bigalpha2026/candidates/hf/hf_002.py",
    "HF-003": "src/bigalpha2026/candidates/hf/hf_002.py",
    "HF-004": "src/bigalpha2026/candidates/hf/hf_002.py",
    "HF-104": ("src/bigalpha2026/candidates/hf/hf_002.py|src/bigalpha2026/candidates/hf/hf_104.py"),
    "OB-001": "src/bigalpha2026/candidates/ob/ob_001.py",
    "OB-002": "src/bigalpha2026/candidates/ob/ob_002.py",
    "OB-003": ("src/bigalpha2026/candidates/ob/ob_001.py|src/bigalpha2026/candidates/ob/ob_002.py"),
    "OB-004": "src/bigalpha2026/candidates/ob/ob_001.py",
    "OB-005": "src/bigalpha2026/candidates/ob/ob_001.py",
    "OB-008": ("src/bigalpha2026/candidates/ob/ob_001.py|src/bigalpha2026/candidates/ob/ob_008.py"),
}


def _literal_assignments(path: Path) -> dict[str, object]:
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
        except (ValueError, TypeError):
            continue
    return values


def _candidate_metadata() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(CANDIDATES.glob("*/*.py")):
        values = _literal_assignments(path)
        candidate_id = values.get("CANDIDATE_ID")
        if not isinstance(candidate_id, str):
            continue
        result[candidate_id] = {
            "component_column": values.get("COMPONENT_COLUMN", ""),
            "source_research_id": values.get("SOURCE_RESEARCH_ID", ""),
        }
    return result


def _source_prefix(source_research_id: object) -> str:
    if not isinstance(source_research_id, str) or not source_research_id:
        return ""
    return source_research_id.split("-", maxsplit=1)[0]


def audit_provenance() -> list[dict[str, object]]:
    metadata = _candidate_metadata()
    results: list[dict[str, object]] = []
    for eligibility in scan_candidates():
        candidate_id = str(eligibility["candidate_id"])
        inputs = list(eligibility["required_inputs"])
        details = metadata.get(candidate_id, {})
        source_id = details.get("source_research_id", "")
        source_prefix = _source_prefix(source_id)
        direct_sources = sorted(
            {
                ALLOWED_DIRECT_INPUTS[input_name]
                for input_name in inputs
                if input_name in ALLOWED_DIRECT_INPUTS
            }
        )
        generator = ""
        if "daily_features" not in inputs:
            status = "verified_direct_allowed_inputs"
            evidence_level = "executable_candidate"
            traced_sources = direct_sources
        elif candidate_id in IN_REPOSITORY_DAILY_FEATURE_GENERATORS:
            status = "verified_repository_generator"
            evidence_level = "executable_upstream_generator"
            generator = IN_REPOSITORY_DAILY_FEATURE_GENERATORS[candidate_id]
            traced_sources = sorted(set(direct_sources) | {"bar1m"})
        elif source_prefix in GENERATOR_BY_SOURCE_PREFIX:
            status = "verified_repository_generator"
            evidence_level = "executable_upstream_generator"
            generator = GENERATOR_BY_SOURCE_PREFIX[source_prefix]
            traced_sources = sorted(set(direct_sources) | {"bar1m"})
        elif source_prefix == "GTJA":
            status = "formula_documented_generator_missing"
            evidence_level = "formula_only"
            traced_sources = sorted(set(direct_sources) | {"bar1m"})
        else:
            status = "unresolved_precomputed_daily_features"
            evidence_level = "candidate_wrapper_only"
            traced_sources = direct_sources

        generator_paths = [item for item in generator.split("|") if item]
        generator_files_exist = bool(generator_paths) and all(
            (ROOT / item).is_file() for item in generator_paths
        )
        if status == "verified_repository_generator" and not generator_files_exist:
            status = "unresolved_missing_generator_file"
            evidence_level = "candidate_wrapper_only"
        results.append(
            {
                "candidate_id": candidate_id,
                "eligible_half_year_static": bool(eligibility["eligible"]),
                "required_inputs": inputs,
                "component_column": details.get("component_column", ""),
                "source_research_id": source_id,
                "traced_sources": traced_sources,
                "generator": generator,
                "generator_files_exist": generator_files_exist,
                "provenance_status": status,
                "evidence_level": evidence_level,
                "source": eligibility["source"],
            }
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "reports/candidate_source_provenance.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "reports/candidate_source_provenance.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = audit_provenance()
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["provenance_status"])
        counts[status] = counts.get(status, 0) + 1
    report = {
        "schema_version": "candidate-source-provenance-v1",
        "policy": {
            "factor_inputs": ["bar1m", "financial"],
            "auxiliary_inputs": ["instruments"],
        },
        "candidate_count": len(rows),
        "status_counts": counts,
        "fully_executable_provenance_count": sum(
            row["evidence_level"] in {"executable_candidate", "executable_upstream_generator"}
            for row in rows
        ),
        "formula_only_count": sum(row["evidence_level"] == "formula_only" for row in rows),
        "missing_executable_generator_count": sum(
            row["evidence_level"] not in {"executable_candidate", "executable_upstream_generator"}
            for row in rows
        ),
        "unresolved_count": sum(
            str(row["provenance_status"]).startswith("unresolved") for row in rows
        ),
        "candidates": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0])
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "required_inputs": "|".join(row["required_inputs"]),
                    "traced_sources": "|".join(row["traced_sources"]),
                }
            )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "fully_executable_provenance_count",
                    "formula_only_count",
                    "missing_executable_generator_count",
                    "unresolved_count",
                    "status_counts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
