#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/projects/bigquant-default
PYTHON=/root/autodl-tmp/conda-envs/quant/bin/python
RAW=/root/bigquant_private_data/bigalpha_2026_stock_bar1m_private_20250101_20260828

cd "$ROOT"

echo "[verify] private bar1m data $(date --iso-8601=seconds)"
"$PYTHON" experiments/finals_pre/common/verify_private1m.py \
  "$RAW" \
  --output reports/dependencies/finals_pre/private1m_transfer_audit.json

echo "[start] progressive additions $(date --iso-8601=seconds)"
progressive_out=reports/dependencies/finals_pre/e3_progressive_add/o2c
mkdir -p "$progressive_out"
find "$progressive_out" -mindepth 1 -maxdepth 1 -type d -name 'p*_seed*' -exec rm -rf {} +
find "$progressive_out" -maxdepth 1 -type f -name 'p*_seed*.log' -delete
rm -f "$progressive_out"/a4_*.csv "$progressive_out"/score_audit.json
bash experiments/finals_pre/e3_progressive_add/run.sh
echo "[done] progressive additions $(date --iso-8601=seconds)"

echo "[start] 60/20 private walk-forward $(date --iso-8601=seconds)"
rm -rf reports/dependencies/finals_pre/e1_o2c_walkforward/model
rm -f reports/dependencies/finals_pre/e1_o2c_walkforward/a4.json \
  reports/dependencies/finals_pre/e1_o2c_walkforward/block_a4.csv \
  reports/dependencies/finals_pre/e1_o2c_walkforward/daily_rank_ic.csv \
  reports/dependencies/finals_pre/e1_o2c_walkforward/rank_ic_by_block.svg
bash experiments/finals_pre/e1_o2c_walkforward/run.sh
echo "[done] 60/20 private walk-forward $(date --iso-8601=seconds)"

echo "[start] epoch curve $(date --iso-8601=seconds)"
epoch_out=reports/dependencies/finals_pre/e2_o2c_epoch_curve
mkdir -p "$epoch_out"
find "$epoch_out" -mindepth 1 -maxdepth 1 -type d -name 'y2024_seed*' -exec rm -rf {} +
rm -f "$epoch_out"/a4_per_seed_epoch.csv "$epoch_out"/a4_summary_epoch.csv
bash experiments/finals_pre/e2_o2c_epoch_curve/run.sh
echo "[done] epoch curve $(date --iso-8601=seconds)"

echo "[start] raw-field direct channels $(date --iso-8601=seconds)"
raw_out=reports/dependencies/finals_pre/e5_raw23_direct/o2c_autodl
mkdir -p "$raw_out"
find "$raw_out" -mindepth 1 -maxdepth 1 -type d \
  \( -name 'baseline17_seed*' -o -name 'raw40_seed*' \) -exec rm -rf {} +
find "$raw_out" -maxdepth 1 -type f -name '*_seed*.log' -delete
rm -f "$raw_out"/training_all_done "$raw_out"/a4_*.csv "$raw_out"/score_audit.json
bash experiments/finals_pre/e5_raw23_direct/run_e5_autodl.sh
"$PYTHON" experiments/finals_pre/e5_raw23_direct/score_autodl.py \
  --source-dir src \
  --output-dir "$raw_out" \
  --labels experiments/finals_pre/e5_raw23_direct/o2c_labels.parquet \
  --exposures /root/autodl-tmp/exposure_2024_full.parquet
echo "[done] raw-field direct channels $(date --iso-8601=seconds)"

echo "[start] frozen private OOS $(date --iso-8601=seconds)"
frozen_out=reports/dependencies/finals_pre/e4_private_fixed_oos
rm -f "$frozen_out"/m_raw_frozen_private_oos.parquet \
  "$frozen_out"/inference_audit.json \
  "$frozen_out"/a4_periods.csv \
  "$frozen_out"/turnover_cost_backtest.csv \
  "$frozen_out"/turnover_cost_daily.parquet \
  "$frozen_out"/score_audit.json
bash experiments/finals_pre/e4_private_fixed_oos/run.sh
echo "[done] frozen private OOS $(date --iso-8601=seconds)"

echo "[all experiments done] $(date --iso-8601=seconds)"
