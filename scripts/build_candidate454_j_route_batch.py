"""Build auditable full-year route files used by Candidate454 joint J runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_submission_j_stability import file_sha256, normalized_route


def combine_route_parts(parts: tuple[Path, ...], output: Path) -> dict[str, object]:
    route = normalized_route(
        pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    route.to_parquet(output, index=False)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "sources": [
            {"path": str(path), "sha256": file_sha256(path)} for path in parts
        ],
        "rows": len(route),
        "days": int(route["date"].nunique()),
        "date_min": str(route["date"].min().date()),
        "date_max": str(route["date"].max().date()),
        "duplicate_keys": int(route.duplicated(["date", "instrument"]).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "reports/dependencies/m_raw_final_checkpoint/combinations_v1/"
            "moe/txm_rolling_unseen_routes"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "reports/candidate454_joint_j_20260803_v2/route_batch"
        ),
    )
    args = parser.parse_args()

    rows = []
    for gate in ("linear", "mlp"):
        for seed in (20260801, 20260802, 20260803):
            parts = tuple(
                args.source_dir / f"{gate}_{seed}_2024_{half}.parquet"
                for half in ("h1", "h2")
            )
            missing = [str(path) for path in parts if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"MoE route parts are missing: {missing}")
            output = args.output_dir / f"moe_{gate}_{seed}_2024.parquet"
            rows.append(
                {
                    "route": f"moe_{gate}_{seed}",
                    **combine_route_parts(parts, output),
                }
            )

    manifest = {
        "protocol": "candidate454_joint_j_route_batch_v1",
        "note": "Only concatenates frozen H1/H2 outputs; no refit or reranking.",
        "routes": rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
