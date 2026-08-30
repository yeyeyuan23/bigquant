"""Export private one-minute bars as a resumable Parquet dataset."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bigquant import dai

TABLE = "bigalpha_2026_stock_bar1m_private"
START = pd.Timestamp(os.environ.get("BQ_PRIVATE1M_START", "2025-01-01"))
END = pd.Timestamp(os.environ.get("BQ_PRIVATE1M_END", "2026-08-29"))
OUTPUT = Path(
    os.environ.get(
        "BQ_PRIVATE1M_OUT",
        "/home/aiuser/work/bigalpha_2026_stock_bar1m_private_20250101_20260828",
    )
)
MANIFEST = OUTPUT / "manifest.json"

FLOAT_COLUMNS = (
    "adjust_factor",
    "pre_close",
    "high",
    "open",
    "low",
    "close",
    "amount",
    "ask_price1",
    "ask_price2",
    "ask_price3",
    "ask_price4",
    "ask_price5",
    "bid_price1",
    "bid_price2",
    "bid_price3",
    "bid_price4",
    "bid_price5",
)
INTEGER_COLUMNS = (
    "deal_number",
    "volume",
    "ask_volume1",
    "ask_volume2",
    "ask_volume3",
    "ask_volume4",
    "ask_volume5",
    "bid_volume1",
    "bid_volume2",
    "bid_volume3",
    "bid_volume4",
    "bid_volume5",
    "ask_num_orders1",
    "ask_num_orders2",
    "ask_num_orders3",
    "ask_num_orders4",
    "ask_num_orders5",
    "bid_num_orders1",
    "bid_num_orders2",
    "bid_num_orders3",
    "bid_num_orders4",
    "bid_num_orders5",
)
SCHEMA = pa.schema(
    [
        pa.field("date", pa.timestamp("ns")),
        pa.field("instrument", pa.string()),
        pa.field("instrument_id", pa.int16()),
        *(pa.field(column, pa.float32()) for column in FLOAT_COLUMNS[:6]),
        pa.field("deal_number", pa.int32()),
        pa.field("volume", pa.int32()),
        pa.field("amount", pa.float32()),
        *(pa.field(column, pa.float32()) for column in FLOAT_COLUMNS[7:]),
        *(pa.field(column, pa.int32()) for column in INTEGER_COLUMNS[2:]),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundaries() -> list[pd.Timestamp]:
    values = list(pd.date_range(START, END, freq="7D"))
    if not values or values[0] != START:
        values.insert(0, START)
    if values[-1] != END:
        values.append(END)
    return values


def normalize_types(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SCHEMA.names).difference(frame.columns))
    if missing:
        raise KeyError(f"missing source columns: {missing}")
    frame = frame[list(SCHEMA.names)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype("string")
    frame["instrument_id"] = pd.to_numeric(
        frame["instrument_id"], errors="coerce"
    ).astype("Int16")
    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    for column in INTEGER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int32")
    return frame.sort_values(["date", "instrument"], kind="mergesort")


def validated_part(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows <= 0 or parquet.schema_arrow != SCHEMA:
        return None
    return {
        "file": path.name,
        "rows": parquet.metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def export_part(start: pd.Timestamp, end: pd.Timestamp, path: Path) -> dict[str, object]:
    start_text = start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")
    frame = dai.query(
        f"""
        SELECT *
        FROM {TABLE}
        WHERE date >= TIMESTAMP '{start_text}' AND date < TIMESTAMP '{end_text}'
        """,
        filters={"date": [start_text, end_text]},
    ).df()
    if frame.empty:
        return {
            "file": path.name,
            "rows": 0,
            "bytes": 0,
            "sha256": None,
        }
    frame = normalize_types(frame)
    table = pa.Table.from_pandas(frame, schema=SCHEMA, preserve_index=False, safe=True)
    partial = path.with_suffix(".parquet.partial")
    partial.unlink(missing_ok=True)
    pq.write_table(
        table,
        partial,
        compression="zstd",
        use_dictionary=["instrument"],
        write_statistics=True,
        row_group_size=262_144,
    )
    if pq.ParquetFile(partial).metadata.num_rows != len(frame):
        raise RuntimeError(f"row mismatch for {path.name}")
    os.replace(partial, path)
    return {
        "file": path.name,
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "min_date": pd.Timestamp(frame["date"].min()).isoformat(),
        "max_date": pd.Timestamp(frame["date"].max()).isoformat(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, object]] = []
    for start, end in pairwise(boundaries()):
        name = f"part_{start:%Y%m%d}_{end - pd.Timedelta(days=1):%Y%m%d}.parquet"
        path = OUTPUT / name
        record = validated_part(path)
        if record is None:
            record = export_part(start, end, path)
        record["query_start"] = start.date().isoformat()
        record["query_end_exclusive"] = end.date().isoformat()
        parts.append(record)
        print(
            f"{name}: rows={int(record['rows']):,}; "
            f"total={sum(int(item['rows']) for item in parts):,}",
            flush=True,
        )
    manifest = {
        "table": TABLE,
        "requested_start": START.date().isoformat(),
        "requested_end_exclusive": END.date().isoformat(),
        "schema": str(SCHEMA),
        "columns": SCHEMA.names,
        "parts": parts,
        "rows": sum(int(item["rows"]) for item in parts),
        "bytes": sum(int(item["bytes"]) for item in parts),
    }
    temporary = MANIFEST.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, MANIFEST)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
