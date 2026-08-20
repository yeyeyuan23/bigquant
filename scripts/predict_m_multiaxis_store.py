"""Replay the frozen full-history M-multiaxis checkpoint on the local E2E store."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "work" / "m_multiaxis"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from unified_m_multiaxis_model import (
    RAW_FIELDS,
    load_checkpoint,
    pack_multiaxis_day,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def standardized_day(path: Path, stats: dict[str, object]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "key", *RAW_FIELDS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"store partition is missing columns: {missing}")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    matrix = frame.loc[:, list(RAW_FIELDS)].to_numpy(np.float32)
    matrix = (matrix - mean[None, :]) / std[None, :]
    matrix[~np.isfinite(matrix)] = 0.0
    frame.loc[:, list(RAW_FIELDS)] = matrix
    return frame


def partition_days(store: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    days = []
    for path in sorted((store / "data").glob("trade_date=*/part.parquet")):
        day = pd.Timestamp(path.parent.name.split("=", 1)[1]).normalize()
        if start <= day <= end:
            days.append(day)
    if not days:
        raise FileNotFoundError("no M E2E partitions in the requested period")
    return days


def run(args: argparse.Namespace) -> None:
    manifest = json.loads((args.store / "manifest.json").read_text())
    if tuple(manifest.get("raw_fields", ())) != RAW_FIELDS:
        raise ValueError("M E2E store does not match the checkpoint raw-field contract")
    mapping = pd.read_csv(args.mapping_csv, dtype={"instrument": str})
    if mapping.duplicated("instrument_id").any() or mapping.duplicated("instrument").any():
        raise ValueError("instrument mapping must be one-to-one")
    id_to_instrument = mapping.set_index("instrument_id")["instrument"].to_dict()

    payload = load_payload(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, stats = load_checkpoint(payload, device=device)
    torch.manual_seed(int(payload.get("seed", 20260804)))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(payload.get("seed", 20260804)))
    days = partition_days(
        args.store,
        pd.Timestamp(args.start).normalize(),
        pd.Timestamp(args.end).normalize(),
    )
    rows = []
    audits = []
    started = time.time()
    for index, day in enumerate(days, start=1):
        path = args.store / "data" / f"trade_date={day.date()}" / "part.parquet"
        frame = standardized_day(path, stats)
        keys = tuple(sorted(map(int, frame["key"].dropna().unique())))
        unmapped = [key for key in keys if key not in id_to_instrument]
        if unmapped:
            raise ValueError(f"{len(unmapped)} store keys are absent from instrument mapping")
        batch = pack_multiaxis_day(
            frame,
            keys=keys,
            stats=stats,
            max_minutes=model.config.max_minutes,
            tail_minutes=model.config.tail_minutes,
            axis_bins=model.config.axis_bins,
        )
        available = np.flatnonzero(batch.stock_mask[0])
        if len(available) < 2:
            raise RuntimeError(f"fewer than two available stocks on {day.date()}")
        tensors = (
            torch.from_numpy(batch.tail_values[:, available]).to(device),
            torch.from_numpy(batch.tail_mask[:, available]).to(device),
            torch.from_numpy(batch.axis_values[:, available]).to(device),
            torch.from_numpy(batch.axis_mask[:, available]).to(device),
            torch.from_numpy(batch.stock_mask[:, available]).to(device),
        )
        with torch.inference_mode():
            prediction = model(*tensors).squeeze(0).float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite M-multiaxis prediction on {day.date()}")
        factor = pd.Series(prediction).rank(method="average", pct=True) * 2.0 - 1.0
        selected_keys = np.asarray(keys)[available]
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "instrument": [id_to_instrument[int(key)] for key in selected_keys],
                    "factor": factor.to_numpy(dtype=np.float64),
                }
            )
        )
        audits.append(
            {
                "date": str(day.date()),
                "store_keys": len(keys),
                "available_keys": len(available),
            }
        )
        if index % 25 == 0 or index == len(days):
            print(
                f"prediction_days={index}/{len(days)} elapsed_seconds={time.time()-started:.1f}",
                flush=True,
            )

    route = pd.concat(rows, ignore_index=True).sort_values(["date", "instrument"])
    if route.empty or route.duplicated(["date", "instrument"]).any():
        raise RuntimeError("M-multiaxis route violates the unique nonempty contract")
    if not np.isfinite(route["factor"].to_numpy()).all():
        raise RuntimeError("M-multiaxis route contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    route.to_parquet(args.output, index=False)
    audit_path = args.output.with_suffix(".days.json")
    audit_path.write_text(json.dumps(audits, indent=2) + "\n")
    report = {
        "status": "complete",
        "route": str(args.output),
        "route_sha256": sha256(args.output),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "store_manifest_sha256": sha256(args.store / "manifest.json"),
        "rows": len(route),
        "days": int(route["date"].nunique()),
        "date_min": str(route["date"].min().date()),
        "date_max": str(route["date"].max().date()),
        "elapsed_seconds": round(time.time() - started, 3),
        "evidence_boundary": (
            "Pure replay of a checkpoint trained on 2019-2024; this route includes "
            "training dates and is not strict OOS or an official score."
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--end", default="2024-12-30")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
