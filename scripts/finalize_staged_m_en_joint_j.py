"""Run preliminary M-multiaxis J, then the full two-EN augmented J."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

CORE_ROUTE_ROOT = Path(
    "reports/dependencies/candidate454_joint_full_history_2019_2024_20260804/route_batch"
)
CORE_ROUTES = {
    "en454_existing_causal": "EN454",
    "x_mlp_frozen": "X",
    "x_tree_frozen": "X_tree",
    "t_frozen": "T",
    "t_residual_frozen": "T_res",
    "m_raw_frozen": "M_raw",
}
NEW_EN_ROUTES = {
    "pema_full454": "EN454_positive_ema",
    "clean_pema": "EN454_clean_positive_ema",
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
    m_manifest = require_complete(args.m_route.with_suffix(".manifest.json"))
    en_manifest = None
    if args.stage == "full":
        if args.en_dir is None:
            raise ValueError("full stage requires --en-dir")
        en_manifest = require_complete(args.en_dir / "manifest.json")

    report_root = args.repo / "reports" / "dependencies" / args.report_name
    stage_dir = report_root / args.stage
    route_dir = stage_dir / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    m_target = route_dir / "m_multiaxis_full.parquet"
    shutil.copy2(args.m_route, m_target)
    shutil.copy2(
        args.m_route.with_suffix(".manifest.json"),
        stage_dir / "m_multiaxis_manifest.json",
    )

    route_specs = [
        {
            "name": name,
            "family": family,
            "parts": [str(CORE_ROUTE_ROOT / f"{name}.parquet")],
        }
        for name, family in CORE_ROUTES.items()
    ]
    route_specs.append(
        {
            "name": "m_multiaxis_full",
            "family": "M_multiaxis_full_history",
            "parts": [str(m_target.relative_to(args.repo))],
        }
    )
    if args.stage == "full":
        assert args.en_dir is not None
        shutil.copy2(args.en_dir / "manifest.json", stage_dir / "en_manifest.json")
        for name, family in NEW_EN_ROUTES.items():
            source = args.en_dir / f"{name}.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = route_dir / source.name
            shutil.copy2(source, target)
            route_specs.append(
                {
                    "name": name,
                    "family": family,
                    "parts": [str(target.relative_to(args.repo))],
                }
            )

    route_names = [spec["name"] for spec in route_specs]
    manifest = {
        "schema_version": 1,
        "protocol": f"staged_m_multiaxis_en454_joint_j_20260805_{args.stage}_v1",
        "evidence_boundary": (
            "Local joint Candidate454 proxy. EN454 routes are causal OOS after their "
            "warmup; frozen X/T/M raw and M-multiaxis include checkpoint training "
            "dates. This is not an official platform score."
        ),
        "old_m_policy": (
            "Only m_raw_frozen is retained among old M routes; M-L5, M residual, "
            "and old M blends are excluded."
        ),
        "routes": route_specs,
        "groups": [
            {
                "name": f"staged_{args.stage}_2019_2024",
                "purpose": (
                    "Preliminary M-multiaxis pool" if args.stage == "preliminary"
                    else "Full pool after P-EMA and Clean-P-EMA completion"
                ),
                "years": [2019, 2020, 2021, 2022, 2023, 2024],
                "routes": route_names,
            }
        ],
    }
    manifest_path = stage_dir / "joint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    joint_dir = stage_dir / "joint_j"
    subprocess.run(
        [
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
        ],
        cwd=args.repo,
        check=True,
    )
    summary_path = joint_dir / "joint_official_proxy_ab_details.csv"
    summary = pd.read_csv(summary_path)
    if set(summary["route"]) != set(route_names) or len(summary) != len(route_names):
        raise RuntimeError("joint J summary route set does not match the stage manifest")
    formula_error = (
        summary["J"] - (0.3 * summary["A"] + 0.7 * summary["B"])
    ).abs().max()
    if float(formula_error) > 1e-12:
        raise RuntimeError(f"joint J formula error is {formula_error}")
    ranked = summary.sort_values("J", ascending=False)
    lines = [
        f"# Staged {args.stage} M-multiaxis and EN454 joint J",
        "",
        "Only M raw is retained among the old M routes. M-multiaxis remains a new",
        "candidate. This local proxy includes training dates for frozen X/T/M routes",
        "and must not be reported as an official platform score.",
        "",
        "| Rank | Route | J | A | B | IC | ICIR | Sharpe | Stress IR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.route}` | {row.J:.6f} | {row.A:.6f} | "
            f"{row.B:.6f} | {row.a_rank_ic_mean:.6f} | {row.a_rank_ic_ir:.6f} | "
            f"{row.a_long_short_sharpe:.6f} | {row.a_stress_ic_ir:.6f} |"
        )
    (stage_dir / "README.md").write_text("\n".join(lines) + "\n")
    completion = {
        "status": "complete",
        "stage": args.stage,
        "route_count": len(route_names),
        "routes": route_names,
        "top_route": str(ranked.iloc[0]["route"]),
        "formula_max_abs_error": float(formula_error),
        "summary_sha256": sha256(summary_path),
        "m_route_sha256": sha256(m_target),
        "m_evidence_boundary": m_manifest.get("evidence_boundary"),
        "en_evidence_boundary": (
            en_manifest.get("evidence_boundary") if en_manifest else None
        ),
    }
    (stage_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    print(json.dumps(completion, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preliminary", "full"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--m-route", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path)
    parser.add_argument("--candidate454-store", type=Path, required=True)
    parser.add_argument(
        "--report-name", default="en454_m_multiaxis_staged_joint_20260805"
    )
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
