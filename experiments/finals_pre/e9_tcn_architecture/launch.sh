#!/usr/bin/env bash
set -euo pipefail
python3 - <<'CHECK'
import json,time
from pathlib import Path
root=Path("./reports/dependencies/finals_pre/tcn_architecture_o2c")
pid=json.loads((root/"status.json").read_text())["pid"]
command=Path(f"/proc/{pid}/cmdline")
while command.exists() and command.read_bytes():
    time.sleep(5)
status=json.loads((root/"status.json").read_text())
if status["state"] != "preflight_complete":
    raise RuntimeError(f"preflight not complete: {status}")
print("Preflight complete; starting all 45 fresh training runs",flush=True)
CHECK
cd .
exec python3 -u experiments/finals_pre/tcn_architecture_o2c/run.py \
  --data-root data \
  --micro-store artifacts/local/unified_microstructure_store_v2_2019_2024 \
  --exposure artifacts/local/exposure_2024_full.parquet \
  --output ./reports/dependencies/finals_pre/tcn_architecture_o2c \
  --device cuda --resume
