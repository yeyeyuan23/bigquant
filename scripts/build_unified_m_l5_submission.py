"""Build the frozen M-v3 L1-L5 AIStudio notebook submission bundle."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import py_compile
import sys
import tempfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import nbformat
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "reports/m_v3_final_2019_2024_seed_20260803"
DEFAULT_SOURCE_ROOT = ROOT / "src/bigalpha2026/alpha_models"
DEFAULT_OUTPUT = ROOT / "submissions/m_l5"
DEFAULT_TRAINING_SCRIPT = ROOT / "scripts/train_unified_microstructure_v3_final.py"
DEFAULT_ROUTE = "M_l5_seed_20260803_final"
UPLOAD_FILES = (
    "unified_m_l5.ipynb",
    "unified_m_l5.py",
    "unified_m_base.py",
    "unified_m_temporal.py",
    "unified_m_microstructure.py",
    "unified_m_microstructure_v2.py",
    "unified_m_l5_checkpoint.py",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_deep_book_sql(table_name: str, start: str, upper: str) -> str:
    """Return the bounded SQL used by the generated online runtime."""

    def positive(side: str, level: int) -> str:
        return (
            f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
            f"THEN {side}_volume{level} ELSE 0 END"
        )

    def sum_depth(side: str, levels: range) -> str:
        return " + ".join(positive(side, level) for level in levels)

    def valid_count(side: str) -> str:
        return " + ".join(
            f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
            "THEN 1 ELSE 0 END"
            for level in range(1, 6)
        )

    bid_l5 = sum_depth("bid", range(1, 6))
    ask_l5 = sum_depth("ask", range(1, 6))
    bid_near = sum_depth("bid", range(1, 3))
    ask_near = sum_depth("ask", range(1, 3))
    valid_bid = valid_count("bid")
    valid_ask = valid_count("ask")
    return f"""
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0 ELSE NULL
                END AS mid_price,
                ({bid_l5}) AS bid_depth_l5,
                ({ask_l5}) AS ask_depth_l5,
                ({bid_near}) AS bid_near_depth,
                ({ask_near}) AS ask_near_depth,
                ({valid_bid}) AS valid_bid_count,
                ({valid_ask}) AS valid_ask_count
            FROM {table_name}
            WHERE date >= TIMESTAMP '{start}'
              AND date < TIMESTAMP '{upper}'
        ),
        derived AS (
            SELECT
                *,
                (bid_depth_l5 - ask_depth_l5)
                    / NULLIF(bid_depth_l5 + ask_depth_l5, 0) AS depth_imbalance_l5,
                bid_near_depth / NULLIF(bid_depth_l5, 0)
                    - ask_near_depth / NULLIF(ask_depth_l5, 0) AS depth_shape_l5
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / NULLIF(lag(mid_price) OVER session_window, 0) - 1.0
                    AS mid_return,
                lead(bid_depth_l5, 5) OVER session_window
                    / NULLIF(bid_depth_l5, 0) - 1.0 AS bid_recovery_5m,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
            WINDOW session_window AS (
                PARTITION BY instrument, trading_day, session_id ORDER BY timestamp
            )
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            avg(CASE WHEN valid_bid_count = 5 AND valid_ask_count = 5
                THEN 1.0 ELSE 0.0 END) AS full_five_levels_rate,
            median(depth_imbalance_l5) AS full_day_depth_imbalance_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance_l5 END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_mid_q10
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery_5m)) END)
                AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(depth_shape_l5) END)
                AS tail_60_shape_sign_consistency
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """


def rewrite_sources(source_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_paths = {
        "frozen_alpha.base": source_root / "base.py",
        "frozen_alpha.temporal": source_root / "temporal.py",
        "frozen_alpha.microstructure": source_root / "microstructure.py",
        "frozen_alpha.microstructure_v2": source_root / "microstructure_v2.py",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source snapshots: {missing}")
    sources = {
        name: path.read_text(encoding="utf-8") for name, path in source_paths.items()
    }
    rewritten = {
        "unified_m_base.py": sources["frozen_alpha.base"],
        "unified_m_temporal.py": "# ruff: noqa: I001\n"
        + sources["frozen_alpha.temporal"].replace(
            "from .base import AlphaModel, register_model",
            "from unified_m_base import AlphaModel, register_model",
        ),
        "unified_m_microstructure.py": "# ruff: noqa: I001\n"
        + sources["frozen_alpha.microstructure"]
        .replace(
            "from .base import AlphaModel, register_model",
            "from unified_m_base import AlphaModel, register_model",
        )
        .replace(
            "from .temporal import DeepSetsContext, MaskedAttentionPool",
            "from unified_m_temporal import DeepSetsContext, MaskedAttentionPool",
        ),
        "unified_m_microstructure_v2.py": "# ruff: noqa: I001\n"
        + sources["frozen_alpha.microstructure_v2"]
        .replace(
            "from .base import AlphaModel, register_model",
            "from unified_m_base import AlphaModel, register_model",
        )
        .replace(
            "from .microstructure import (",
            "from unified_m_microstructure import (",
        )
        .replace(
            "from .temporal import DeepSetsContext, MaskedAttentionPool",
            "from unified_m_temporal import DeepSetsContext, MaskedAttentionPool",
        ),
    }
    source_hashes = {
        name: sha256_bytes(source.encode("utf-8")) for name, source in sources.items()
    }
    return rewritten, source_hashes


ENTRY_TEMPLATE = r'''"""Frozen M-v3 five-level AIStudio entrypoint; inference only."""

FROZEN_MANIFEST = __FROZEN_MANIFEST__


def _load_frozen_model(torch, io, base64, zlib, device):
    from unified_m_l5_checkpoint import CHECKPOINT_B85_ZLIB
    from unified_m_microstructure_v2 import MicrostructureV2Model

    checkpoint_bytes = zlib.decompress(base64.b85decode(CHECKPOINT_B85_ZLIB.encode("ascii")))
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model = MicrostructureV2Model(**payload["config"])
    model.network.load_state_dict(payload["state_dict"], strict=True)
    model.network.to(device)
    model.network.eval()
    return model


def _positive_depth_sql(side, level):
    return (
        f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
        f"THEN {side}_volume{level} ELSE 0 END"
    )


def _sum_depth_sql(side, levels):
    return " + ".join(_positive_depth_sql(side, level) for level in levels)


def _valid_count_sql(side):
    return " + ".join(
        f"CASE WHEN {side}_price{level} > 0 AND {side}_volume{level} > 0 "
        "THEN 1 ELSE 0 END"
        for level in range(1, 6)
    )


def _build_deep_book_sql(table_name, start, upper):
    bid_l5 = _sum_depth_sql("bid", range(1, 6))
    ask_l5 = _sum_depth_sql("ask", range(1, 6))
    bid_near = _sum_depth_sql("bid", range(1, 3))
    ask_near = _sum_depth_sql("ask", range(1, 3))
    valid_bid = _valid_count_sql("bid")
    valid_ask = _valid_count_sql("ask")
    return f"""
        WITH base AS (
            SELECT
                date AS timestamp,
                instrument,
                CAST(date_trunc('day', date) AS DATE) AS trading_day,
                CASE WHEN EXTRACT(hour FROM date) < 12 THEN 0 ELSE 1 END AS session_id,
                CASE
                    WHEN bid_price1 > 0 AND ask_price1 >= bid_price1
                         AND bid_volume1 > 0 AND ask_volume1 > 0
                    THEN (bid_price1 + ask_price1) / 2.0 ELSE NULL
                END AS mid_price,
                ({bid_l5}) AS bid_depth_l5,
                ({ask_l5}) AS ask_depth_l5,
                ({bid_near}) AS bid_near_depth,
                ({ask_near}) AS ask_near_depth,
                ({valid_bid}) AS valid_bid_count,
                ({valid_ask}) AS valid_ask_count
            FROM {table_name}
            WHERE date >= TIMESTAMP '{start}'
              AND date < TIMESTAMP '{upper}'
        ),
        derived AS (
            SELECT
                *,
                (bid_depth_l5 - ask_depth_l5)
                    / NULLIF(bid_depth_l5 + ask_depth_l5, 0) AS depth_imbalance_l5,
                bid_near_depth / NULLIF(bid_depth_l5, 0)
                    - ask_near_depth / NULLIF(ask_depth_l5, 0) AS depth_shape_l5
            FROM base
        ),
        sequenced AS (
            SELECT
                *,
                mid_price / NULLIF(lag(mid_price) OVER session_window, 0) - 1.0
                    AS mid_return,
                lead(bid_depth_l5, 5) OVER session_window
                    / NULLIF(bid_depth_l5, 0) - 1.0 AS bid_recovery_5m,
                row_number() OVER (
                    PARTITION BY instrument, trading_day ORDER BY timestamp DESC
                ) AS reverse_minute
            FROM derived
            WINDOW session_window AS (
                PARTITION BY instrument, trading_day, session_id ORDER BY timestamp
            )
        ),
        thresholds AS (
            SELECT
                *,
                quantile_cont(mid_return, 0.10) OVER (
                    PARTITION BY instrument, trading_day
                ) AS negative_mid_q10
            FROM sequenced
        )
        SELECT
            CAST(trading_day AS DATETIME) AS date,
            instrument,
            avg(CASE WHEN valid_bid_count = 5 AND valid_ask_count = 5
                THEN 1.0 ELSE 0.0 END) AS full_five_levels_rate,
            median(depth_imbalance_l5) AS full_day_depth_imbalance_median,
            median(CASE WHEN reverse_minute <= 60 THEN depth_imbalance_l5 END)
                AS tail_60_bid_depth_imbalance_median,
            median(CASE WHEN mid_return < 0 AND mid_return <= negative_mid_q10
                THEN GREATEST(-2.0, LEAST(2.0, bid_recovery_5m)) END)
                AS negative_mid_shock_q10_bid_depth_recovery_5m_median,
            avg(CASE WHEN reverse_minute <= 60 THEN sign(depth_shape_l5) END)
                AS tail_60_shape_sign_consistency
        FROM thresholds
        GROUP BY trading_day, instrument
        ORDER BY date, instrument
    """


def _iter_bar1m_chunks(dai, pd, table_name, start_ts, end_ts):
    levels = range(1, 6)
    columns = (
        "date", "instrument", "open", "high", "low", "close", "amount", "volume",
        "deal_number",
        *(f"ask_price{level}" for level in levels),
        *(f"bid_price{level}" for level in levels),
        *(f"ask_volume{level}" for level in levels),
        *(f"bid_volume{level}" for level in levels),
    )
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=13), end_ts)
        upper = chunk_end + pd.Timedelta(days=1)
        query = f"""
            SELECT {', '.join(columns)}
            FROM {table_name}
            WHERE date >= TIMESTAMP '{cursor:%Y-%m-%d}'
              AND date < TIMESTAMP '{upper:%Y-%m-%d}'
            ORDER BY date, instrument
        """
        filters = {
            "date": [cursor.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]
        }
        raw = dai.query(query, filters=filters, compression=True).df()
        context = dai.query(
            _build_deep_book_sql(
                table_name,
                cursor.strftime("%Y-%m-%d"),
                upper.strftime("%Y-%m-%d"),
            ),
            filters=filters,
            compression=True,
        ).df()
        if not raw.empty:
            yield raw, context
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_stock_pool(dai, pd, start_ts, end_ts):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")]
        },
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=["date", "instrument"])
    pool = pool.loc[pool["date"].between(start_ts, end_ts)]
    pool = pool.drop_duplicates(["date", "instrument"])
    if pool.empty:
        raise RuntimeError("stock pool produced no rows")
    return {
        pd.Timestamp(day): tuple(sorted(group["instrument"].unique()))
        for day, group in pool.groupby("date", sort=True)
    }


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
    torch.manual_seed(20260803)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_frozen_model(torch, io, base64, zlib, device)
    from unified_m_microstructure_v2 import (
        build_microstructure_v2_features,
        pack_microstructure_v2_days,
    )

    pool_by_day = _load_stock_pool(dai, pd, start_ts, end_ts)
    outputs = []
    for raw_chunk, context_chunk in _iter_bar1m_chunks(
        dai, pd, datasources["bar1m"], start_ts, end_ts
    ):
        raw_chunk["date"] = pd.to_datetime(raw_chunk["date"], errors="coerce")
        raw_chunk["instrument"] = raw_chunk["instrument"].astype(str)
        raw_chunk = raw_chunk.dropna(subset=["date", "instrument"])
        raw_chunk = raw_chunk.sort_values(["date", "instrument"], kind="stable")
        raw_chunk["trade_date"] = raw_chunk["date"].dt.normalize()
        context_chunk["date"] = pd.to_datetime(
            context_chunk["date"], errors="coerce"
        ).dt.normalize()
        context_chunk["instrument"] = context_chunk["instrument"].astype(str)
        context_chunk = context_chunk.dropna(subset=["date", "instrument"])
        if context_chunk.duplicated(["date", "instrument"]).any():
            raise RuntimeError("five-level context contains duplicate keys")
        for day, raw_day in raw_chunk.groupby("trade_date", sort=True):
            instruments = pool_by_day.get(pd.Timestamp(day), ())
            if len(instruments) < 2:
                continue
            raw_day = raw_day.loc[raw_day["instrument"].isin(instruments)]
            deep_day = context_chunk.loc[context_chunk["date"].eq(pd.Timestamp(day))]
            if deep_day.empty:
                raise RuntimeError(f"five-level context is missing for {pd.Timestamp(day).date()}")
            features = build_microstructure_v2_features(
                raw_day.drop(columns="trade_date"),
                deep_day,
            )
            batch = pack_microstructure_v2_days(
                features,
                dates=[day],
                instruments=instruments,
                max_minutes=242,
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
                    {
                        "date": pd.Timestamp(day),
                        "instrument": np.asarray(instruments),
                        "factor": daily_factor,
                    }
                )
            )
            del features, batch, values, observed, minutes, stocks, prediction
        del raw_chunk, context_chunk
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


def build_notebook(path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from unified_m_l5 import main  # noqa: F401\n",
                id="unified-m-l5-entrypoint",
            )
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, path)


def build_handoff(title: str) -> str:
    """Return copy-paste AIStudio commands for a submission subdirectory."""

    return f"""# {title}

Upload exactly the seven files listed in `submission_bundle_manifest.json`
into one directory. The commands default to `/home/aiuser/work/sub_m_l5`;
set `M_L5_DIR` first if your directory has a different name.
The notebook contains one code cell: `from unified_m_l5 import main`.

```bash
M_L5_DIR="${{M_L5_DIR:-/home/aiuser/work/sub_m_l5}}"
M_L5_PROBE="${{M_L5_PROBE:-/home/aiuser/work/aistudio_submission_lookahead_probe.py}}"
cd "$M_L5_DIR"
test -f unified_m_l5.py
test -f "$M_L5_PROBE"

python -m jupyter nbconvert --to notebook --execute unified_m_l5.ipynb --output unified_m_l5_executed.ipynb --ExecutePreprocessor.timeout=180
python -m py_compile unified_m_l5.py unified_m_base.py unified_m_temporal.py unified_m_microstructure.py unified_m_microstructure_v2.py unified_m_l5_checkpoint.py
PYTHONPATH="$M_L5_DIR" python "$M_L5_PROBE" \\
  "$M_L5_DIR/unified_m_l5.py" \\
  --start 2024-12-23 \\
  --cutoff 2024-12-27 \\
  --end 2024-12-30 \\
  --bar1m bigalpha_2026_stock_bar1m \\
  | tee "$M_L5_DIR/unified_m_l5_probe.log"
```

Pass only when status is `ok`, `total_difference_rows=0`, every cutoff has
`prefix_invariant=true`, the output contract has no missing dates or unexpected
keys, and the accepted submission file list contains all seven files.
"""


def smoke_model(output_dir: Path, expected_parameter_count: int) -> dict[str, object]:
    entry_path = output_dir / "unified_m_l5.py"
    sys.path.insert(0, str(output_dir))
    try:
        spec = importlib.util.spec_from_file_location("m_l5_submission_smoke", entry_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import generated M-l5 entrypoint")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module._load_frozen_model(torch, io, base64, zlib, torch.device("cpu"))
        parameter_count = sum(parameter.numel() for parameter in model.network.parameters())
        if parameter_count != expected_parameter_count:
            raise RuntimeError(
                f"parameter count mismatch: {parameter_count} != {expected_parameter_count}"
            )
        values = torch.randn(1, 3, 8, 28)
        observed = torch.ones_like(values, dtype=torch.bool)
        minutes = torch.ones(1, 3, 8, dtype=torch.bool)
        stocks = torch.ones(1, 3, dtype=torch.bool)
        with torch.inference_mode():
            output = model.predict((values, observed, minutes, stocks))
        if tuple(output.shape) != (1, 3) or not torch.isfinite(output).all():
            raise RuntimeError("generated checkpoint smoke inference failed")
        return {
            "parameter_count": parameter_count,
            "output_shape": list(output.shape),
            "finite": True,
        }
    finally:
        sys.path.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-tree-dirty", action="store_true")
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument(
        "--training-script", type=Path, default=DEFAULT_TRAINING_SCRIPT
    )
    parser.add_argument("--handoff-title")
    parser.add_argument(
        "--evidence-boundary",
        default=(
            "Frozen final checkpoint and local static validation only; "
            "AIStudio execution, prefix evidence, and platform score remain separate."
        ),
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.training_script = args.training_script.resolve()

    checkpoint = args.run_dir / "unified_microstructure_v3_final_checkpoint.pt"
    checkpoint_manifest_path = args.run_dir / "final_checkpoint_manifest.json"
    if not checkpoint.is_file() or not checkpoint_manifest_path.is_file():
        raise FileNotFoundError("final checkpoint or manifest is missing")
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != checkpoint_manifest.get("checkpoint_sha256"):
        raise ValueError("final checkpoint SHA-256 does not match its manifest")
    config = checkpoint_manifest.get("config", {})
    if config.get("input_dim") != 28:
        raise ValueError("M-l5 final checkpoint must have exactly 28 input channels")

    rewritten, source_hashes = rewrite_sources(args.source_root)
    checkpoint_payload = base64.b85encode(
        zlib.compress(checkpoint.read_bytes(), level=9)
    ).decode("ascii")
    frozen_manifest = {
        "route": args.route,
        "model": "unified_microstructure_v3",
        "training_commit": args.training_commit,
        "training_tree_dirty": args.training_tree_dirty,
        "training_protocol": checkpoint_manifest["training_protocol"],
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_training_window": [
            checkpoint_manifest["training_start"],
            checkpoint_manifest["training_end"],
        ],
        "label_isolation_gap_days": checkpoint_manifest["label_isolation_gap_days"],
        "seed": checkpoint_manifest["seed"],
        "input_schema": {
            "profile": "canonical_dynamic_l1_l5",
            "max_minutes": config["max_minutes"],
            "channels": config["input_dim"],
            "book_levels": [1, 2, 3, 4, 5],
        },
        "allowed_sources": ["bar1m", "stock_pool"],
        "bar1m_table": "bigalpha_2026_stock_bar1m",
        "stock_pool_table": "bigalpha_2026_instruments",
        "parameter_count": checkpoint_manifest["parameter_count"],
        "training_script_sha256": sha256_file(args.training_script),
        "source_sha256": source_hashes,
        "evidence_boundary": args.evidence_boundary,
    }
    if checkpoint_manifest.get("prediction_start"):
        frozen_manifest["local_oos_evidence"] = {
            key: checkpoint_manifest[key]
            for key in (
                "prediction_start",
                "prediction_end",
                "prediction_days",
                "missing_prediction_days",
                "rank_ic_mean",
                "candidate454_J",
                "candidate454_A",
                "candidate454_B",
                "candidate454_B_model_score",
                "candidate454_B_mean_abs_weight",
                "candidate454_B_std_abs_weight",
            )
            if key in checkpoint_manifest
        }
    module_sources = {
        **rewritten,
        "unified_m_l5_checkpoint.py": (
            '"""Frozen M-l5 checkpoint payload; generated, do not edit."""\n\n'
            f"CHECKPOINT_B85_ZLIB = {checkpoint_payload!r}\n"
        ),
        "unified_m_l5.py": ENTRY_TEMPLATE.replace(
            "__FROZEN_MANIFEST__", repr(frozen_manifest)
        ),
    }
    for filename, source in module_sources.items():
        compile(source, filename, "exec")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    module_paths: dict[str, Path] = {}
    for filename, source in module_sources.items():
        path = args.output_dir / filename
        path.write_text(source, encoding="utf-8")
        module_paths[filename] = path
    notebook_path = args.output_dir / "unified_m_l5.ipynb"
    build_notebook(notebook_path)
    with tempfile.TemporaryDirectory(prefix="unified-m-l5-build-") as compile_dir:
        for filename, path in module_paths.items():
            py_compile.compile(
                str(path),
                cfile=str(Path(compile_dir) / f"{filename}c"),
                doraise=True,
            )
    smoke = smoke_model(args.output_dir, int(frozen_manifest["parameter_count"]))

    manifest_path = args.output_dir / "unified_m_l5_manifest.json"
    manifest_path.write_text(json.dumps(frozen_manifest, indent=2) + "\n", encoding="utf-8")
    bundle = {
        "schema_version": 1,
        "notebook": "unified_m_l5.ipynb",
        "notebook_code": "from unified_m_l5 import main  # noqa: F401\n",
        "upload_file_count": len(UPLOAD_FILES),
        "upload_files": [
            {
                "name": filename,
                "bytes": (args.output_dir / filename).stat().st_size,
                "sha256": sha256_file(args.output_dir / filename),
            }
            for filename in UPLOAD_FILES
        ],
        "checkpoint_manifest": str(checkpoint_manifest_path.relative_to(ROOT)),
    }
    (args.output_dir / "submission_bundle_manifest.json").write_text(
        json.dumps(bundle, indent=2) + "\n",
        encoding="utf-8",
    )
    validation = {
        "generated_at": datetime.now(UTC).isoformat(),
        "build": "passed",
        "py_compile": "passed",
        "notebook_structure": "one_code_cell",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_smoke": smoke,
        "online_deep_book_source": "bounded_bar1m_l1_l5_sql",
        "platform_output_contract": "pending_aistudio",
        "platform_prefix_probe": "pending_aistudio",
    }
    (args.output_dir / "unified_m_l5_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    handoff_title = args.handoff_title or f"{args.route} AIStudio handoff"
    handoff = build_handoff(handoff_title)
    (args.output_dir / "aistudio_handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"bundle": bundle, "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
