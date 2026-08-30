"""Verify a transferred private one-minute Parquet dataset against its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        record["file"]: record
        for record in manifest["parts"]
        if int(record["rows"]) > 0
    }
    actual = {path.name: path for path in args.dataset.glob("part_*.parquet")}
    missing = sorted(set(expected).difference(actual))
    unexpected = sorted(set(actual).difference(expected))
    failures: list[str] = []
    rows = 0
    bytes_total = 0
    for index, name in enumerate(sorted(expected), start=1):
        path = actual.get(name)
        if path is None:
            continue
        record = expected[name]
        parquet = pq.ParquetFile(path)
        file_rows = parquet.metadata.num_rows
        file_bytes = path.stat().st_size
        rows += file_rows
        bytes_total += file_bytes
        if file_rows != int(record["rows"]):
            failures.append(f"{name}:rows")
        if file_bytes != int(record["bytes"]):
            failures.append(f"{name}:bytes")
        if parquet.schema_arrow.names != manifest["columns"]:
            failures.append(f"{name}:columns")
        if sha256(path) != record["sha256"]:
            failures.append(f"{name}:sha256")
        if index % 10 == 0:
            print(f"[verify] files={index}/{len(expected)}", flush=True)
    result = {
        "table": manifest.get("table"),
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "rows": rows,
        "expected_rows": int(manifest["rows"]),
        "bytes": bytes_total,
        "expected_bytes": int(manifest["bytes"]),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if (
        missing
        or unexpected
        or failures
        or rows != int(manifest["rows"])
        or bytes_total != int(manifest["bytes"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
