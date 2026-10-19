"""Verify one E5 Parquet day against its source NPZ tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    channels = [str(value) for value in manifest["channels"]]
    minutes = int(manifest["max_minutes"])
    with np.load(args.npz, allow_pickle=False) as payload:
        expected_values = payload["values"].astype(np.float32, copy=False)
        expected_instruments = payload["instruments"].astype(str)

    table = pq.read_table(args.parquet)
    actual_instruments = np.asarray(table["instrument"].to_pylist(), dtype=str)
    channel_values = []
    for channel in channels:
        column = table[channel].combine_chunks()
        child = column.values.to_numpy(zero_copy_only=False)
        channel_values.append(child.reshape(table.num_rows, minutes))
    actual_values = np.stack(channel_values, axis=-1).astype(np.float32, copy=False)

    finite_expected = np.isfinite(expected_values)
    finite_actual = np.isfinite(actual_values)
    finite_match = bool(np.array_equal(finite_expected, finite_actual))
    compared = finite_expected & finite_actual
    absolute_error = np.abs(expected_values[compared] - actual_values[compared])
    result = {
        "expected_shape": list(expected_values.shape),
        "actual_shape": list(actual_values.shape),
        "row_count": table.num_rows,
        "trade_dates": sorted({str(value) for value in table["trade_date"].to_pylist()}),
        "instrument_match": bool(np.array_equal(expected_instruments, actual_instruments)),
        "finite_mask_match": finite_match,
        "finite_count": int(compared.sum()),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "nonzero_error_count": int(np.count_nonzero(absolute_error)),
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    if (
        result["expected_shape"] != result["actual_shape"]
        or not result["instrument_match"]
        or not result["finite_mask_match"]
        or result["nonzero_error_count"] != 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
