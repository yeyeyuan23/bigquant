"""Build the frozen raw-microstructure AIStudio submission pair."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import py_compile
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "src/alpha_models"
DEFAULT_OUTPUT = ROOT / "submissions/m_raw_240"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build_sources(
    checkpoint: Path,
    source_root: Path,
    *,
    expected_checkpoint_sha256: str,
    training_commit: str,
    training_tree_dirty: bool,
    checkpoint_block: int,
    training_start: str,
    training_end: str,
    training_protocol: str,
) -> tuple[dict[str, str], dict[str, object]]:
    import torch

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha256}"
        )
    checkpoint_config = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )["config"]
    if checkpoint_config["input_dim"] != 17:
        raise ValueError("M_raw checkpoint must have 17 input channels")
    if checkpoint_config["max_minutes"] != 240:
        raise ValueError("current export requires a 240-minute checkpoint; use the frozen bundle for historical weights")
    source_paths = {
        f"frozen_alpha.{name}": (
            source_root / f"{name}.py"
            if (source_root / f"{name}.py").is_file()
            else source_root / f"alpha_models_{name}.py"
        )
        for name in ("base", "temporal", "microstructure")
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source snapshots: {missing}")
    frozen_sources = {
        name: path.read_text(encoding="utf-8")
        for name, path in source_paths.items()
    }
    source_sha256 = {
        name: sha256_bytes(source.encode("utf-8"))
        for name, source in frozen_sources.items()
    }
    encoded_checkpoint = base64.b85encode(zlib.compress(checkpoint_bytes, level=9)).decode(
        "ascii"
    )
    frozen_manifest = {
        "route": "M_raw",
        "model": "unified_microstructure",
        "training_commit": training_commit,
        "training_tree_dirty": bool(training_tree_dirty),
        "training_protocol": training_protocol,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_block": int(checkpoint_block),
        "checkpoint_training_window": [training_start, training_end],
        "input_schema": {
            "profile": "canonical",
            "max_minutes": int(checkpoint_config["max_minutes"]),
            "channels": 17,
            "book_levels": [1, 2, 3],
        },
        "allowed_sources": ["bar1m", "stock_pool"],
        "stock_pool_table": "bigalpha_2026_instruments",
        "source_sha256": source_sha256,
        "evidence_boundary": (
            "Frozen checkpoint and static validation only; AIStudio execution, "
            "prefix evidence, and platform score remain separate."
        ),
    }
    base_source = frozen_sources["frozen_alpha.base"]
    temporal_source = frozen_sources["frozen_alpha.temporal"].replace(
        "from .base import AlphaModel, register_model",
        "from unified_m_base import AlphaModel, register_model",
    )
    microstructure_source = frozen_sources["frozen_alpha.microstructure"]
    microstructure_source = microstructure_source.replace(
        "from .base import AlphaModel, register_model",
        "from unified_m_base import AlphaModel, register_model",
    ).replace(
        "from .temporal import DeepSetsContext, MaskedAttentionPool",
        "from unified_m_temporal import DeepSetsContext, MaskedAttentionPool",
    )
    weights_source = f'''"""Frozen M-raw checkpoint payload; generated, do not edit."""

CHECKPOINT_B85_ZLIB = {encoded_checkpoint!r}
'''
    entry_source = f'''"""Frozen M-raw AIStudio entrypoint; inference only."""

FROZEN_MANIFEST = {frozen_manifest!r}


def _load_frozen_model(torch, io, base64, zlib, device):
    from unified_m_checkpoint import CHECKPOINT_B85_ZLIB
    from unified_m_microstructure import MicrostructureModel

    checkpoint_bytes = zlib.decompress(base64.b85decode(CHECKPOINT_B85_ZLIB.encode("ascii")))
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model = MicrostructureModel(**payload["config"])
    model.network.load_state_dict(payload["state_dict"], strict=True)
    model.network.to(device)
    model.network.eval()
    return model


def _iter_bar1m_chunks(dai, pd, table_name, start_ts, end_ts):
    columns = (
        "date", "instrument", "open", "high", "low", "close", "amount", "volume",
        "deal_number", "ask_price1", "ask_price2", "ask_price3", "bid_price1",
        "bid_price2", "bid_price3", "ask_volume1", "ask_volume2", "ask_volume3",
        "bid_volume1", "bid_volume2", "bid_volume3",
    )
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=13), end_ts)
        upper = chunk_end + pd.Timedelta(days=1)
        query = f"""
            SELECT {{', '.join(columns)}}
            FROM {{table_name}}
            WHERE date >= TIMESTAMP '{{cursor:%Y-%m-%d}}'
              AND date < TIMESTAMP '{{upper:%Y-%m-%d}}'
            ORDER BY date, instrument
        """
        frame = dai.query(
            query,
            filters={{
                "date": [
                    cursor.strftime("%Y-%m-%d"),
                    upper.strftime("%Y-%m-%d"),
                ]
            }},
            compression=True,
        ).df()
        if not frame.empty:
            yield frame
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_stock_pool(dai, pd, start_ts, end_ts):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={{
            "date": [
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
            ]
        }},
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"])
    pool = pool.loc[pool["date"].between(start_ts, end_ts)]
    pool = pool.drop_duplicates(["date", "instrument"])
    if pool.empty:
        raise RuntimeError("stock pool produced no rows")
    return {{
        pd.Timestamp(day): tuple(sorted(group["instrument"].unique()))
        for day, group in pool.groupby("date", sort=True)
    }}


def main(datasources, start_date, end_date):
    import base64
    import io
    import zlib

    import dai
    import numpy as np
    import pandas as pd
    import torch

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date precedes start_date")
    if "bar1m" not in datasources:
        raise KeyError("datasources must contain bar1m")
    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_frozen_model(torch, io, base64, zlib, device)
    from unified_m_microstructure import (
        build_microstructure_features,
        pack_microstructure_days,
    )

    pool_by_day = _load_stock_pool(dai, pd, start_ts, end_ts)
    outputs = []
    for raw_chunk in _iter_bar1m_chunks(
        dai, pd, datasources["bar1m"], start_ts, end_ts
    ):
        raw_chunk["date"] = pd.to_datetime(raw_chunk["date"], errors="coerce")
        raw_chunk["instrument"] = raw_chunk["instrument"].astype(str)
        raw_chunk = raw_chunk.dropna(subset=["date", "instrument"])
        raw_chunk = raw_chunk.sort_values(["date", "instrument"], kind="stable")
        raw_chunk["trade_date"] = raw_chunk["date"].dt.normalize()
        for day, raw_day in raw_chunk.groupby("trade_date", sort=True):
            instruments = pool_by_day.get(pd.Timestamp(day), ())
            if len(instruments) < 2:
                continue
            raw_day = raw_day.loc[raw_day["instrument"].isin(instruments)]
            features = build_microstructure_features(raw_day.drop(columns="trade_date"))
            batch = pack_microstructure_days(
                features,
                dates=[day],
                instruments=instruments,
                max_minutes=model.config.max_minutes,
            )
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 2:
                continue
            values = torch.from_numpy(batch.values[:, available]).to(device)
            observed = torch.from_numpy(batch.observed_mask[:, available]).to(device)
            minutes = torch.from_numpy(batch.minute_mask[:, available]).to(device)
            stocks = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            with torch.inference_mode():
                prediction = (
                    model.predict((values, observed, minutes, stocks))
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            factor = pd.Series(prediction).rank(method="average", pct=True).to_numpy()
            factor = factor * 2.0 - 1.0
            daily_factor = np.zeros(len(instruments), dtype=np.float32)
            daily_factor[available] = factor.astype(np.float32)
            outputs.append(
                pd.DataFrame(
                    {{
                        "date": pd.Timestamp(day),
                        "instrument": np.asarray(instruments),
                        "factor": daily_factor,
                    }}
                )
            )
            del features, batch, values, observed, minutes, stocks, prediction
        del raw_chunk
    if not outputs:
        raise RuntimeError("bar1m produced no valid prediction rows")
    result = pd.concat(outputs, ignore_index=True)
    result = result.loc[result["date"].between(start_ts, end_ts)]
    result = result.sort_values(["date", "instrument"], kind="stable").reset_index(drop=True)
    if result.empty:
        raise RuntimeError("factor output is empty")
    if result.duplicated(["date", "instrument"]).any():
        raise RuntimeError("factor output contains duplicate date/instrument keys")
    if not np.isfinite(result["factor"].to_numpy()).all():
        raise RuntimeError("factor output contains non-finite values")
    if result.groupby("date")["factor"].nunique().le(1).any():
        raise RuntimeError("factor output contains a constant daily cross-section")
    return result[["date", "instrument", "factor"]]
'''
    module_sources = {
        "unified_m_raw.py": entry_source,
        "unified_m_base.py": base_source,
        "unified_m_temporal.py": temporal_source,
        "unified_m_microstructure.py": microstructure_source,
        "unified_m_checkpoint.py": weights_source,
    }
    for filename, module_source in module_sources.items():
        compile(module_source, filename, "exec")
    return module_sources, frozen_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-checkpoint-sha256", required=True
    )
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-tree-dirty", action="store_true")
    parser.add_argument("--checkpoint-block", type=int, required=True)
    parser.add_argument("--training-start", required=True)
    parser.add_argument("--training-end", required=True)
    parser.add_argument("--training-protocol", required=True)
    args = parser.parse_args()
    module_sources, manifest = build_sources(
        args.checkpoint,
        args.source_root,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        training_commit=args.training_commit,
        training_tree_dirty=args.training_tree_dirty,
        checkpoint_block=args.checkpoint_block,
        training_start=args.training_start,
        training_end=args.training_end,
        training_protocol=args.training_protocol,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module_paths = {}
    for filename, module_source in module_sources.items():
        module_path = args.output_dir / filename
        module_path.write_text(module_source, encoding="utf-8")
        module_paths[filename] = module_path
    with tempfile.TemporaryDirectory(prefix="unified-m-build-") as compile_dir:
        for filename, module_path in module_paths.items():
            py_compile.compile(
                str(module_path),
                cfile=str(Path(compile_dir) / f"{filename}c"),
                doraise=True,
            )
    stale_notebook = args.output_dir / "unified_m_raw.ipynb"
    if stale_notebook.exists():
        stale_notebook.unlink()
    manifest_path = args.output_dir / "unified_m_raw_manifest.json"
    report_path = args.output_dir / "unified_m_raw_validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "entrypoint": str(module_paths["unified_m_raw.py"]),
        "modules": {
            filename: {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for filename, path in module_paths.items()
        },
        "manifest": str(manifest_path),
        "submission_format": "modular_python",
        "upload_files": list(module_paths),
        "validation": {
            "build": "passed",
            "py_compile": "passed",
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "platform": "pending_new_checkpoint_validation",
            "prefix": "pending_new_checkpoint_validation",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
