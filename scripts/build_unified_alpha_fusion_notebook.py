"""Build the source-coupled Unified Alpha Fusion development notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notebooks" / "unified_alpha_fusion_development.ipynb"


def build_notebook(
    *,
    candidate_pool: Path,
    candidate_manifest: Path,
    microstructure_store: Path,
) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Unified Alpha Fusion Development\n\n"
            "This notebook is generated from the development repository and "
            "imports the canonical implementation from "
            "`src/bigalpha2026/alpha_models`; it does not contain a handwritten "
            "second model. T uses a fixed 60-day sample sequence; the Elastic Net "
            "baseline and M use 60-day training / one-day label isolation / "
            "20-day prediction blocks. The upstream "
            "artifact must contain all 462 candidates through 2024 before the "
            "formal suite starts."
        ),
        nbformat.v4.new_code_cell(
            f"""import json
import sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from bigalpha2026.alpha_models import (
    DEFAULT_SUBMISSION_DATA_CONTRACT,
    MICROSTRUCTURE_CHANNELS,
    ModelFactory,
    candidate_ids_from_manifest,
)

EXPECTED_CANDIDATES = 462
CANDIDATE_POOL = Path({str(candidate_pool)!r})
CANDIDATE_MANIFEST = Path({str(candidate_manifest)!r})
MICROSTRUCTURE_STORE = Path({str(microstructure_store)!r})
manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="utf-8"))
candidate_ids = candidate_ids_from_manifest(CANDIDATE_MANIFEST)
date_range = manifest.get("date_range")
candidate_ready = (
    CANDIDATE_POOL.exists()
    and len(candidate_ids) == EXPECTED_CANDIDATES
    and isinstance(date_range, list)
    and len(date_range) == 2
    and str(date_range[1]) >= "2024-12-31"
)
print({{
    "candidate_count": len(candidate_ids),
    "expected_candidate_count": EXPECTED_CANDIDATES,
    "candidate_ready": candidate_ready,
    "date_range": date_range,
    "candidate_pool": str(CANDIDATE_POOL),
}})
micro_manifest_path = MICROSTRUCTURE_STORE / "manifest.json"
micro_ready = False
if micro_manifest_path.is_file():
    micro_manifest = json.loads(micro_manifest_path.read_text(encoding="utf-8"))
    micro_ready = tuple(micro_manifest.get("channels", ())) == MICROSTRUCTURE_CHANNELS
print({{
    "microstructure_ready": micro_ready,
    "microstructure_store": str(MICROSTRUCTURE_STORE),
    "required_channels": len(MICROSTRUCTURE_CHANNELS),
}})"""
        ),
        nbformat.v4.new_markdown_cell("## Contracts"),
        nbformat.v4.new_code_cell(
            """contract = DEFAULT_SUBMISSION_DATA_CONTRACT
print({
    "allowed_factor_sources": ["bar1m", "financial"],
    "training_start": str(contract.training_start),
    "training_end": str(contract.training_end),
    "temporal_lookback_days": contract.temporal_lookback_days,
    "candidate_feature_count": EXPECTED_CANDIDATES,
})"""
        ),
        nbformat.v4.new_markdown_cell("## T: Factor Temporal"),
        nbformat.v4.new_code_cell(
            """adapter = ModelFactory.create(
    "unified_temporal",
    {
        "input_dim": EXPECTED_CANDIDATES,
        "model_dim": 256,
        "lookback": 60,
        "kernels": (3, 5, 15),
        "transformer_layers": 4,
        "attention_heads": 8,
        "feedforward_dim": 768,
        "dropout": 0.12,
    },
)
model = adapter.network
print({
    "registered_model": "unified_temporal",
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "architecture": "candidate462 temporal CNN+Transformer + DeepSets",
})"""
        ),
        nbformat.v4.new_markdown_cell("## M: Raw Microstructure"),
        nbformat.v4.new_code_cell(
            """micro_adapter = ModelFactory.create(
    "unified_microstructure",
    {
        "model_dim": 96,
        "max_minutes": 242,
        "kernels": (3, 15, 60),
        "tcn_blocks": 3,
        "tail_minutes": 30,
        "dropout": 0.1,
    },
)
micro_model = micro_adapter.network
print({
    "registered_model": "unified_microstructure",
    "parameters": sum(parameter.numel() for parameter in micro_model.parameters()),
    "architecture": "raw minute multi-scale TCN + explicit statistics + DeepSets",
    "training_status": "not started" if not micro_ready else "data ready; training not run",
})"""
        ),
        nbformat.v4.new_markdown_cell("## Formal unified experiment command"),
        nbformat.v4.new_code_cell(
            """if not candidate_ready:
    print("BLOCKED: upstream artifact is not candidate462-complete")
else:
    print("bash run_unified_alpha_fusion_suite.sh")
    print({
        "baseline": "candidate462 full-pool Elastic Net, 60d train / 20d predict",
        "routes": [
            "unified_temporal",
            "unified_mlp",
            "unified_lightgbm",
            *( ["unified_microstructure"] if micro_ready else [] ),
        ],
        "microstructure": "ready" if micro_ready else "skipped: canonical store missing",
        "j_policy": "standalone J + paired delta-J over Elastic Net baseline",
        "training_history": "expanding from 2019",
    })"""
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"unified-{index:02d}"
    nbformat.validate(notebook)
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--microstructure-store",
        type=Path,
        default=Path("/root/autodl-tmp/microstructure_store"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    notebook = build_notebook(
        candidate_pool=args.candidate_pool,
        candidate_manifest=args.candidate_manifest,
        microstructure_store=args.microstructure_store,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
