#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/projects/bigquant-unified-integration
python=/root/autodl-tmp/conda-envs/quant/bin/python
run_root="$repo/reports/xt_protocol_retrain_20260803"

cd "$repo"
mkdir -p "$run_root"
exec "$python" scripts/run_xt_protocol_retrain.py \
  --output-root "$run_root"
