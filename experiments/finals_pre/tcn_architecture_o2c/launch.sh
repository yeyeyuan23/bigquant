#!/usr/bin/env bash
set -euo pipefail
/root/autodl-tmp/conda-envs/quant/bin/python - <<'CHECK'
import json,time
from pathlib import Path
root=Path("/root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/tcn_architecture_o2c")
pid=json.loads((root/"status.json").read_text())["pid"]
command=Path(f"/proc/{pid}/cmdline")
while command.exists() and command.read_bytes():
    time.sleep(5)
status=json.loads((root/"status.json").read_text())
if status["state"] != "preflight_complete":
    raise RuntimeError(f"preflight not complete: {status}")
print("Preflight complete; starting all 45 fresh training runs",flush=True)
CHECK
cd /root/autodl-tmp/projects/bigquant-default
exec /root/autodl-tmp/conda-envs/quant/bin/python -u experiments/finals_pre/tcn_architecture_o2c/run.py \
  --data-root /root/autodl-tmp/data \
  --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
  --exposure /root/autodl-tmp/exposure_2024_full.parquet \
  --output /root/autodl-tmp/projects/bigquant-default/reports/dependencies/finals_pre/tcn_architecture_o2c \
  --device cuda --resume
