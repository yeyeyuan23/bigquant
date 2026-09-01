"""Align the canonical 2023-2024 O2C labels to the E5 Parquet store."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
STORE = Path("/root/bigquant_private_data/e5_raw40_2023_2024_parquet")
OUTPUT = ROOT / "experiments/finals_pre/e5_raw23_direct/o2c_labels.parquet"
AUDIT = ROOT / "experiments/finals_pre/e5_raw23_direct/o2c_labels_audit.json"
LABEL_FILES = (
    Path("/root/autodl-tmp/data/labels/year=2023/part-2023.parquet"),
    Path("/root/autodl-tmp/data/labels/year=2024/part-2024.parquet"),
)
KEYS = ["date", "instrument"]
LABEL = "ret_next_open_to_close"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def store_keys() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    parts = sorted(STORE.glob("day=*.parquet"))
    if len(parts) != 484:
        raise RuntimeError(f"expected 484 E5 day files, found {len(parts)}")
    for path in parts:
        day = pd.Timestamp(path.stem.removeprefix("day="))
        frame = pd.read_parquet(path, columns=["instrument"])
        frame["date"] = day
        rows.append(frame[KEYS])
    keys = pd.concat(rows, ignore_index=True)
    keys["instrument"] = keys["instrument"].astype(str)
    if keys.duplicated(KEYS).any():
        raise RuntimeError("E5 Parquet store contains duplicate date/instrument keys")
    return keys.sort_values(KEYS, kind="stable")


def main() -> int:
    if not (STORE / "export_manifest.json").is_file():
        raise FileNotFoundError(STORE / "export_manifest.json")
    for path in LABEL_FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    keys = store_keys()
    labels = pd.concat(
        [pd.read_parquet(path, columns=[*KEYS, LABEL]) for path in LABEL_FILES],
        ignore_index=True,
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    if labels.duplicated(KEYS).any():
        raise RuntimeError("canonical O2C labels contain duplicate keys")
    aligned = keys.merge(labels, on=KEYS, how="left", validate="one_to_one")
    if len(aligned) != len(keys):
        raise RuntimeError("E5 label alignment changed the store key count")
    temporary = OUTPUT.with_suffix(".parquet.partial")
    aligned.to_parquet(temporary, index=False, compression="zstd")
    check = pd.read_parquet(temporary)
    if len(check) != len(aligned) or check.duplicated(KEYS).any():
        raise RuntimeError("written E5 O2C label artifact failed round-trip validation")
    os.replace(temporary, OUTPUT)
    audit = {
        "label": LABEL,
        "source": "canonical AutoDL labels/year=2023,2024",
        "source_sha256": {str(path): sha256(path) for path in LABEL_FILES},
        "rows": len(aligned),
        "days": int(aligned["date"].nunique()),
        "start": aligned["date"].min().date().isoformat(),
        "end": aligned["date"].max().date().isoformat(),
        "non_null_rows": int(aligned[LABEL].notna().sum()),
        "non_null_2024_days": int(
            aligned.loc[
                (aligned["date"].dt.year == 2024) & aligned[LABEL].notna(), "date"
            ].nunique()
        ),
        "duplicate_keys": int(aligned.duplicated(KEYS).sum()),
        "output_sha256": sha256(OUTPUT),
    }
    AUDIT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
