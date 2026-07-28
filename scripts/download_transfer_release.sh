#!/usr/bin/env bash
set -euo pipefail

tag="${1:-data-transfer-20260728}"
out_dir="${2:-data/transfers}"
manifest="artifacts/transfers/SHA256SUMS"

mkdir -p "${out_dir}"
gh release download "${tag}" --dir "${out_dir}" --clobber

python3 - <<'PY' "${manifest}" "${out_dir}"
import hashlib
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

failures = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, name = line.split(maxsplit=1)
    path = out_dir / name
    if not path.exists():
        continue
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        failures.append(f"{path}: expected {expected}, got {actual}")
    else:
        print(f"ok {path}")

if failures:
    print("checksum failures:", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(3)
PY
