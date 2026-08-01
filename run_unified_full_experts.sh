#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/projects/bigquant-unified-integration"
DATA_ROOT="/root/autodl-tmp/projects/bigquant/data"
PYTHON_BIN="/root/autodl-tmp/conda-envs/quant/bin/python"
STORE_ROOT="/root/autodl-tmp/unified_microstructure_store_v2_2019_2024"
RUN_ID="${UNIFIED_RUN_ID:-20260801_full_experts_v1}"
MICRO_LOG="$PROJECT_ROOT/reports/${RUN_ID}_micro_store.log"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/reports"

if [[ ! -s "$STORE_ROOT/manifest.json" ]]; then
  if [[ -e "$STORE_ROOT" ]]; then
    echo "incomplete microstructure store already exists: $STORE_ROOT" >&2
    exit 1
  fi
  inputs=("$DATA_ROOT"/e2e_parquet/bigalpha_2026_e2e_bar1m/*.parquet)
  if [[ ${#inputs[@]} -ne 72 ]]; then
    echo "expected 72 monthly minute files, found ${#inputs[@]}" >&2
    exit 1
  fi
  "$PYTHON_BIN" scripts/prepare_unified_microstructure_store.py \
    --source-profile e2e_compressed \
    --instrument-map "$DATA_ROOT/e2e_parquet/instrument_id_map_internal_2019_2024.csv" \
    --input "${inputs[@]}" \
    --output-dir "$STORE_ROOT" \
    2>&1 | tee "$MICRO_LOG"
fi

export UNIFIED_MICROSTRUCTURE_STORE="$STORE_ROOT"
export UNIFIED_RUN_ID="$RUN_ID"
exec bash run_unified_alpha_fusion_suite.sh
