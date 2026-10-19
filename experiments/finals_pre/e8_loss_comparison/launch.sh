#!/usr/bin/env bash
set -euo pipefail
cd .
base=./reports/dependencies/finals_pre/e8_loss_comparison
if [ -e "$base/20260915" ]; then
  echo 'E8 output already exists; inspect status and use run.py --resume explicitly.' >&2
  exit 1
fi
mkdir -p "$base"
nohup python3 -u experiments/finals_pre/e8_loss_comparison/run.py \
  --data-root data \
  --micro-store artifacts/local/unified_microstructure_store_v2_2019_2024 \
  --exposure artifacts/local/exposure_2024_full.parquet \
  --output "$base/20260915" --device cuda \
  > "$base/20260915.launch.log" 2>&1 < /dev/null &
printf 'E8 driver PID: %s\nLog: %s\n' "$!" "$base/20260915.launch.log"
