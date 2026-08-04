"""Build the three-file upload package from a frozen multi-axis checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import torch

MODEL_IMPORT_BLOCK = """from unified_m_multiaxis_model import (
    RAW_FIELDS,
    MultiAxisConfig,
    MultiAxisNetwork,
    checkpoint_payload,
    load_checkpoint,
    pack_multiaxis_day,
    transform_raw_frame,
)
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_training_entrypoint(source: str) -> str:
    source = source.replace("from __future__ import annotations\n", "")
    if MODEL_IMPORT_BLOCK not in source:
        raise RuntimeError("training source no longer has the expected model import block")
    source = source.replace(MODEL_IMPORT_BLOCK, "")
    source = source.replace("import numpy as np\n", "")
    source = source.replace("import pandas as pd\n", "")
    source = source.replace("import torch\n", "")
    marker = '\n\nif __name__ == "__main__":\n'
    if marker not in source:
        raise RuntimeError("training source no longer has the expected CLI entrypoint")
    return source.rsplit(marker, 1)[0].strip()


def _submission_train_source(source_dir: Path) -> str:
    model_source = (source_dir / "unified_m_multiaxis_model.py").read_text().rstrip()
    training_source = _without_training_entrypoint(
        (source_dir / "train_m_multiaxis.py").read_text()
    )
    weight_runtime = r'''

import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def _deserialize_state(payload, map_location):
    device = torch.device(map_location) if isinstance(map_location, str) else map_location
    return {
        name: torch.tensor(item["data"], dtype=getattr(torch, item["dtype"]))
        .reshape(item["shape"])
        .to(device)
        for name, item in payload.items()
    }


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as stream:
        serialized = json.load(stream)
    device = torch.device(map_location) if isinstance(map_location, str) else map_location
    payload = {
        "config": serialized["config"],
        "raw_fields": serialized["raw_fields"],
        "axis_names": serialized["axis_names"],
        "preprocessing": serialized["preprocessing"],
        "seed": serialized["seed"],
        "training": serialized["training"],
        "state_dict": _deserialize_state(serialized["state_dict"], device),
    }
    return load_checkpoint(payload, device=device)
'''
    return model_source + "\n\n" + training_source + "\n" + weight_runtime.strip() + "\n"


def _serialize_state(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
    serialized = {}
    for name, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        serialized[name] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }
    return serialized


def _write_weights(checkpoint: Path, output: Path) -> dict[str, object]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    required = {
        "config", "raw_fields", "axis_names", "preprocessing", "seed", "training", "state_dict"
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"checkpoint is missing keys: {missing}")
    serialized = {
        "schema_version": 2,
        "model_kind": "tail60_time_volatility_turnover_cross_attention",
        "raw_fields": payload["raw_fields"],
        "axis_names": payload["axis_names"],
        "config": payload["config"],
        "preprocessing": payload["preprocessing"],
        "seed": payload["seed"],
        "training": payload["training"],
        "state_dict": _serialize_state(payload["state_dict"]),
    }
    output.write_text(json.dumps(serialized, ensure_ascii=False, separators=(",", ":")))
    return serialized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.py"
    train_path.write_text(_submission_train_source(args.source_dir))
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(train_path)], check=True)
    weights = _write_weights(args.checkpoint, args.output_dir / "weights.json")
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# BigAlpha 2026 multi-axis M\nTail-60 main path with chronological time, volatility, and turnover axes."
        ),
        nbformat.v4.new_code_cell((args.source_dir / "predict_m_multiaxis.py").read_text()),
    ]
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbformat.write(notebook, args.output_dir / "predict.ipynb")
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output_dir.iterdir()) if path.is_file()
    ]
    audit = {
        "status": "built",
        "upload_files": ["predict.ipynb", "train.py", "weights.json"],
        "checkpoint_sha256": sha256(args.checkpoint),
        "raw_fields": weights["raw_fields"],
        "axes": weights["axis_names"],
        "output_columns": ["date", "instrument", "score"],
        "training": weights["training"],
        "files": files,
        "evidence_boundary": "Local package only; AIStudio execution and official score are separate.",
    }
    if args.audit_report is not None:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
