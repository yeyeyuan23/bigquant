#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/projects/bigquant-candidate454-completion"
PYTHON_BIN="/root/autodl-tmp/conda-envs/quant/bin/python"
DATA_ROOT="/root/autodl-tmp/projects/bigquant/data"
WORK_ROOT="${BIGALPHA_FULL_FACTOR_WORK:-/root/autodl-tmp/candidate454_completion_full_2019_2024}"
LOG_ROOT="$WORK_ROOT/logs"

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

export BIGALPHA_PROJECT_ROOT="/root/autodl-tmp/projects/bigquant"
export BIGALPHA_E2E_BAR1M_DIR="$DATA_ROOT/e2e_parquet/bigalpha_2026_e2e_bar1m"
export BIGALPHA_INSTRUMENT_MAP="$DATA_ROOT/e2e_parquet/instrument_id_map_internal_2019_2024.csv"
export BIGALPHA_FACTOR_WIKI_WORK="$WORK_ROOT"
export BIGALPHA_FACTOR_WIKI_YEARS="2019,2020,2021,2022,2023,2024"

"$PYTHON_BIN" scripts/factor_wiki_remaining/build_daily_base.py \
  2>&1 | tee "$LOG_ROOT/daily_base.log"

"$PYTHON_BIN" scripts/factor_wiki_remaining/build_haitong_components.py \
  >"$LOG_ROOT/haitong.log" 2>&1 &
HAITONG_PID=$!

BIGALPHA_FACTOR_WIKI_WORKERS=8 \
  "$PYTHON_BIN" scripts/factor_wiki_remaining/build_changjiang_components.py \
  >"$LOG_ROOT/changjiang.log" 2>&1 &
CHANGJIANG_PID=$!

wait "$HAITONG_PID"
wait "$CHANGJIANG_PID"

"$PYTHON_BIN" scripts/factor_wiki_remaining/build_remaining132_feature_matrix.py \
  --work "$WORK_ROOT" \
  --candidate-pool "$DATA_ROOT/runtime/all156_full_2019_2024_corrected/factors/candidate_pool.parquet" \
  --period-tag 2019_2024 \
  --output-dir "$WORK_ROOT/remaining132" \
  2>&1 | tee "$LOG_ROOT/remaining132.log"

echo "[$(date '+%F %T')] remaining132 2019-2024 complete: $WORK_ROOT"
