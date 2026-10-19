"""Convert selected monthly Feather members from an official E2E ZIP to Parquet.

The archive is processed one member at a time so extracted Feather files do not
accumulate on disk. Outputs are atomic and accompanied by a conversion manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from pyarrow import feather, parquet

MONTH_RE = re.compile(r"(?P<year>20\d{2})(?P<month>0[1-9]|1[0-2])\.0\.feather$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--password-env", default="BIGALPHA_E2E_ZIP_PASSWORD")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_members(
    archive: zipfile.ZipFile,
    years: set[int],
) -> list[zipfile.ZipInfo]:
    members = []
    for info in archive.infolist():
        match = MONTH_RE.search(Path(info.filename).name)
        if match and int(match.group("year")) in years:
            members.append(info)
    members.sort(key=lambda info: Path(info.filename).name)
    if not members:
        raise FileNotFoundError(
            f"no monthly Feather members found for years={sorted(years)}"
        )
    return members


def convert_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    output_dir: Path,
    password: bytes | None,
    compression_level: int,
    overwrite: bool,
) -> dict[str, object]:
    output = output_dir / f"{Path(info.filename).stem}.parquet"
    if output.exists() and not overwrite:
        metadata = parquet.read_metadata(output)
        return {
            "member": info.filename,
            "archive_crc32": f"{info.CRC:08x}",
            "output": str(output),
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "bytes": output.stat().st_size,
            "status": "reused_existing",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e2e_feather_", dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        feather_path = temp_root / Path(info.filename).name
        parquet_path = temp_root / output.name
        with archive.open(info, pwd=password) as source, feather_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
        table = feather.read_table(feather_path, memory_map=True)
        parquet.write_table(
            table,
            parquet_path,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        metadata = parquet.read_metadata(parquet_path)
        if metadata.num_rows != table.num_rows or metadata.num_columns != table.num_columns:
            raise ValueError(f"Parquet metadata mismatch for {info.filename}")
        os.replace(parquet_path, output)
        schema = [f"{field.name}:{field.type}" for field in table.schema]

    return {
        "member": info.filename,
        "archive_crc32": f"{info.CRC:08x}",
        "output": str(output),
        "rows": metadata.num_rows,
        "columns": metadata.num_columns,
        "schema": schema,
        "bytes": output.stat().st_size,
        "status": "converted",
    }


def main() -> int:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    password_text = os.environ.get(args.password_env)
    password = password_text.encode("utf-8") if password_text else None
    years = set(args.year)
    results = []
    with zipfile.ZipFile(args.archive) as archive:
        members = selected_members(archive, years)
        for index, info in enumerate(members, start=1):
            result = convert_member(
                archive,
                info,
                output_dir=args.output_dir,
                password=password,
                compression_level=args.compression_level,
                overwrite=args.overwrite,
            )
            results.append(result)
            print(
                json.dumps(
                    {"progress": f"{index}/{len(members)}", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    manifest = {
        "archive": str(args.archive),
        "archive_bytes": args.archive.stat().st_size,
        "years": sorted(years),
        "outputs": results,
    }
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
