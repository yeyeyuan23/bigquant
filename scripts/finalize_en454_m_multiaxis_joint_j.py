"""Materialize and score the three EN454 experiments with full M-multiaxis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROUTES = {
    "b0_full454": "EN454_baseline",
    "pema_full454": "EN454_positive_ema",
    "clean_pema": "EN454_clean_positive_ema",
    "m_multiaxis_full": "M_multiaxis_full_history",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_complete(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"artifact is not complete: {path}")
    return payload


def run(args: argparse.Namespace) -> None:
    en_manifest = require_complete(args.en_dir / "manifest.json")
    m_manifest = require_complete(args.m_route.with_suffix(".manifest.json"))
    destination = args.repo / "reports" / "dependencies" / args.report_name
    route_dir = destination / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "b0_full454": args.en_dir / "b0_full454.parquet",
        "pema_full454": args.en_dir / "pema_full454.parquet",
        "clean_pema": args.en_dir / "clean_pema.parquet",
        "m_multiaxis_full": args.m_route,
    }
    copied = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = route_dir / f"{name}.parquet"
        shutil.copy2(source, target)
        copied[name] = {
            "path": str(target.relative_to(args.repo)),
            "sha256": sha256(target),
        }
    shutil.copy2(args.en_dir / "manifest.json", destination / "en_manifest.json")
    shutil.copy2(
        args.m_route.with_suffix(".manifest.json"), destination / "m_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "protocol": "en454_stability_plus_m_multiaxis_joint_j_20260805_v1",
        "evidence_boundary": (
            "EN routes are continuous causal OOS after the warmup. M-multiaxis is "
            "a full-history checkpoint replay that includes training dates. The joint J "
            "is a local proxy, not an official platform score."
        ),
        "routes": [
            {
                "name": name,
                "family": family,
                "parts": [copied[name]["path"]],
            }
            for name, family in ROUTES.items()
        ],
        "groups": [
            {
                "name": "en454_stability_m_multiaxis_2019_2024",
                "purpose": (
                    "Compressed three-route EN454 stability decision with the frozen "
                    "full-history M-multiaxis route in one joint Candidate454 pool."
                ),
                "years": [2019, 2020, 2021, 2022, 2023, 2024],
                "routes": list(ROUTES),
            }
        ],
    }
    manifest_path = destination / "joint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    joint_dir = destination / "joint_j"
    command = [
        str(args.python),
        str(args.repo / "scripts" / "run_candidate454_joint_v2_recalc.py"),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(joint_dir),
        "--data-dir",
        str(args.repo / "data"),
        "--reports-dir",
        str(args.repo / "reports"),
        "--candidate454-store",
        str(args.candidate454_store),
    ]
    subprocess.run(command, cwd=args.repo, check=True)
    summary_path = joint_dir / "joint_official_proxy_ab_details.csv"
    summary = pd.read_csv(summary_path)
    if set(summary["route"]) != set(ROUTES) or len(summary) != len(ROUTES):
        raise RuntimeError("joint J summary does not contain exactly the four routes")
    formula_error = (
        summary["J"] - (0.3 * summary["A"] + 0.7 * summary["B"])
    ).abs().max()
    if float(formula_error) > 1e-12:
        raise RuntimeError(f"joint J formula error is {formula_error}")
    ranked = summary.sort_values("J", ascending=False)
    readme = [
        "# EN454 stability and M-multiaxis joint J",
        "",
        "EN routes are continuous causal OOS after a 60-day warmup. The M route",
        "replays a checkpoint trained on the full 2019-2024 period, so the joint J",
        "is in-sample-inclusive for M and is not an official platform score.",
        "",
        "| Rank | Route | J | A | B | IC | ICIR | Sharpe | Stress IR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        readme.append(
            f"| {rank} | `{row.route}` | {row.J:.6f} | {row.A:.6f} | "
            f"{row.B:.6f} | {row.a_rank_ic_mean:.6f} | {row.a_rank_ic_ir:.6f} | "
            f"{row.a_long_short_sharpe:.6f} | {row.a_stress_ic_ir:.6f} |"
        )
    (destination / "README.md").write_text("\n".join(readme) + "\n")
    completion = {
        "status": "complete",
        "report": str(destination),
        "routes": copied,
        "en_manifest_sha256": sha256(destination / "en_manifest.json"),
        "m_manifest_sha256": sha256(destination / "m_manifest.json"),
        "joint_manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "formula_max_abs_error": float(formula_error),
        "top_route": str(ranked.iloc[0]["route"]),
        "source_evidence": {
            "en": en_manifest.get("evidence_boundary"),
            "m": m_manifest.get("evidence_boundary"),
        },
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path, required=True)
    parser.add_argument("--m-route", type=Path, required=True)
    parser.add_argument("--candidate454-store", type=Path, required=True)
    parser.add_argument(
        "--report-name", default="en454_stability_m_multiaxis_joint_20260805"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
