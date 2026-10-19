#!/usr/bin/env bash
set -euo pipefail

repo="${BIGQUANT_ROOT:-.}"
python="${PYTHON_BIN:-python3}"
run_root="$repo/reports/xt_protocol_retrain_20260803"

cd "$repo"
mkdir -p "$run_root"
exec "$python" scripts/run_xt_protocol_retrain.py \
  --output-root "$run_root"
