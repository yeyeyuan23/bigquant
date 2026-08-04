"""Build the frozen original Temporal-deep AIStudio submission."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import py_compile
import re
import sys
import zlib
from pathlib import Path

import torch

ROOT = Path("/root/autodl-tmp/projects/bigquant-unified-integration")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_unified_full_boosting_submission as shared

SUITE = ROOT / "reports/unified_alpha_fusion_suite_20260801_full_experts_v1"
OUTPUT = ROOT / "submissions/t"
T_CHECKPOINT = (
    SUITE / "temporal_deep_2024" / "unified_temporal_2024_h2_checkpoint.pt"
)
EXPECTED_T_SHA256 = "6ded4fe2877bfbf413d9d7aaa38e1f6fb2bd97fd6e9f98edfac33f315e77bcb2"
MAX_FILE_BYTES = 50 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def temporal_runtime_source() -> str:
    source = (ROOT / "src/bigalpha2026/alpha_models/temporal.py").read_text()
    source = re.sub(r"^from __future__ import annotations\n", "", source, flags=re.MULTILINE)
    source = source.replace("from .base import AlphaModel, register_model\n", "")
    source = source.split('@register_model("unified_temporal")', 1)[0]
    return "\n\n# ---- frozen Temporal runtime ----\n" + source


def candidate_spec() -> dict[str, object]:
    payload = torch.load(
        io.BytesIO(shared.load_x_checkpoint()),
        map_location="cpu",
        weights_only=True,
    )
    spec = payload.get("candidate454_spec")
    if not isinstance(spec, dict) or len(spec.get("candidate_ids", ())) != 454:
        raise RuntimeError("frozen X metadata does not contain Candidate454")
    return spec


ENTRY_TEMPLATE = r'''"""Frozen original Temporal-deep expert for AIStudio."""

from __future__ import annotations

import base64
import hashlib
import io
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

KEY_COLUMNS = ("date", "instrument")
FROZEN_CANDIDATE_SPEC = __CANDIDATE_SPEC__
FROZEN_MANIFEST = {
    "route": "__T_ROUTE__",
    "model": "temporal_deep",
    "checkpoint_sha256": "__T_SHA__",
    "checkpoint_training_end": "__TRAINING_END__",
    "selection_evidence": "__SELECTION_EVIDENCE__",
    "training": False,
    "history_calendar_days": 183,
    "lookback_trading_days": 60,
    "candidate_count": 454,
    "allowed_sources": ["bar1m", "financial", "stock_pool_keys"],
    "evidence_boundary": "Frozen original T inference; not T-residual and not a platform score.",
}


def _load_checkpoint(torch):
    import zlib
    from unified_t_weights import CHECKPOINT

    compressed = base64.b85decode(b"".join(CHECKPOINT["b85"]))
    if len(compressed) != CHECKPOINT["compressed_bytes"]:
        raise RuntimeError("T compressed checkpoint byte-count mismatch")
    if hashlib.sha256(compressed).hexdigest() != CHECKPOINT["compressed_sha256"]:
        raise RuntimeError("T compressed checkpoint hash mismatch")
    payload = zlib.decompress(compressed)
    if len(payload) != CHECKPOINT["bytes"]:
        raise RuntimeError("T checkpoint byte-count mismatch")
    if hashlib.sha256(payload).hexdigest() != CHECKPOINT["sha256"]:
        raise RuntimeError("T checkpoint hash mismatch")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _rank(values, pd):
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series.notna() & series.abs().lt(float("inf"))
    if int(finite.sum()) < 2:
        raise RuntimeError("T produced fewer than two finite cross-sectional predictions")
    ranked = pd.Series(0.0, index=series.index, dtype="float64")
    ranked.loc[finite] = (
        series.loc[finite].rank(method="average", pct=True).to_numpy(dtype="float64")
        * 2.0
        - 1.0
    )
    return ranked.to_numpy(dtype="float64")


def main(datasources, start_date, end_date):
    import gc
    import numpy as np
    import pandas as pd
    import torch

    if "bar1m" not in datasources or "financial" not in datasources:
        raise KeyError("datasources must contain bar1m and financial")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date precedes start_date")

    torch.manual_seed(20260803)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = _load_checkpoint(torch)
    config = CandidateTemporalConfig(**payload["config"])
    model = CandidateTemporalNetwork(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    candidate_ids = tuple(FROZEN_CANDIDATE_SPEC["candidate_ids"])
    panel = build_candidate454(
        datasources,
        start,
        end,
        candidate_spec=FROZEN_CANDIDATE_SPEC,
        return_history=True,
    )
    if panel.empty or panel.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("Candidate454 history panel is empty or duplicated")
    missing = sorted(set(candidate_ids).difference(panel.columns))
    if missing:
        raise RuntimeError(f"Candidate454 schema mismatch: {missing}")

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    history_days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    requested_days = history_days[(history_days >= start) & (history_days <= end)]
    if len(requested_days) == 0:
        raise RuntimeError("no requested trading dates in Candidate454 panel")
    by_day = {
        day: block.set_index("instrument").sort_index()
        for day, block in panel.groupby("date", sort=True)
    }

    outputs = []
    with torch.inference_mode():
        for day in requested_days:
            current = by_day[pd.Timestamp(day)]
            instruments = tuple(current.index.astype(str))
            position = history_days.get_loc(day)
            window_days = history_days[
                max(0, position - config.lookback + 1) : position + 1
            ]
            if len(window_days) < config.lookback:
                raise RuntimeError(
                    f"insufficient T history for {day.date()}: {len(window_days)} days"
                )
            window = np.stack(
                [
                    by_day[past]
                    .reindex(instruments)
                    .loc[:, list(candidate_ids)]
                    .to_numpy(dtype="float32", copy=True)
                    for past in window_days
                ],
                axis=1,
            )
            values = torch.from_numpy(window).unsqueeze(0).to(device)
            observed = torch.isfinite(values)
            stocks = torch.ones(
                (1, len(instruments)), dtype=torch.bool, device=device
            )
            raw = model(values, observed, stocks).squeeze(0).float().cpu().numpy()
            outputs.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(day),
                        "instrument": instruments,
                        "factor": _rank(raw, pd),
                    }
                )
            )
            del window, values, observed, stocks, raw
            gc.collect()

    output = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(list(KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    output = output.loc[
        output["date"].between(start, end), ["date", "instrument", "factor"]
    ]
    if output.empty or output.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("T output is empty or duplicated")
    if not np.isfinite(output["factor"].to_numpy()).all():
        raise RuntimeError("T output contains non-finite values")
    if output.groupby("date")["factor"].nunique().le(1).any():
        raise RuntimeError("T output contains a constant day")
    return output
'''


def checkpoint_source(payload: bytes) -> str:
    compressed = zlib.compress(payload, level=9)
    return "\n".join(
        [
            '"""Frozen original Temporal-deep checkpoint; generated, do not edit."""',
            "",
            "CHECKPOINT = {",
            f"    'sha256': {sha256_bytes(payload)!r},",
            f"    'bytes': {len(payload)},",
            f"    'compressed_sha256': {sha256_bytes(compressed)!r},",
            f"    'compressed_bytes': {len(compressed)},",
            "    'b85': " + shared.bytes_literal_chunks(compressed).replace("\n", "\n    ") + ",",
            "}",
            "",
        ]
    )


def notebook_source() -> str:
    payload = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "t-protocol-best-intro",
                "metadata": {},
                "source": [
                    "# Frozen original Temporal-deep expert\n",
                    "Upload this notebook with its three sibling Python modules.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "t-protocol-best-entrypoint",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["from unified_t import main\n"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(payload, ensure_ascii=False, indent=1) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=T_CHECKPOINT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_T_SHA256
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--route", default="T_original")
    parser.add_argument("--training-end", default="unknown")
    parser.add_argument("--selection-evidence", default="original frozen T")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.read_bytes()
    checkpoint_sha256 = sha256_bytes(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("original T checkpoint SHA-256 mismatch")
    spec = candidate_spec()
    entry = (
        ENTRY_TEMPLATE.replace("__CANDIDATE_SPEC__", repr(spec))
        .replace("__T_SHA__", checkpoint_sha256)
        .replace("__T_ROUTE__", args.route)
        .replace("__TRAINING_END__", args.training_end)
        .replace("__SELECTION_EVIDENCE__", args.selection_evidence)
    )
    entry += temporal_runtime_source()
    entry += "\n\n# ---- Candidate454 feature runtime ----\n"
    entry += shared.feature_runtime_suffix()
    runtime = shared.merged_candidate_runtime_source()
    ast.parse(entry, filename="unified_t.py")
    ast.parse(runtime, filename="unified_candidate454_runtime.py")

    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        "unified_t.py": entry,
        "unified_t_weights.py": checkpoint_source(checkpoint),
        "unified_candidate454_runtime.py": runtime,
        "unified_t.ipynb": notebook_source(),
    }
    for name, source in files.items():
        (args.output / name).write_text(source, encoding="utf-8")
    for path in args.output.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    oversized = {
        path.name: path.stat().st_size
        for path in args.output.iterdir()
        if path.is_file() and path.stat().st_size >= MAX_FILE_BYTES
    }
    if oversized:
        raise RuntimeError(f"submission file exceeds 50 MiB: {oversized}")
    manifest = {
        "route": args.route,
        "upload_file_count": len(files),
        "files": [
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name in sorted(files)
            for path in [args.output / name]
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_training_end": args.training_end,
        "selection_evidence": args.selection_evidence,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
