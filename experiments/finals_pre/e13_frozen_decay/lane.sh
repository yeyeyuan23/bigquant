#!/usr/bin/env bash
# E13: fixed-cutoff annual decay, current 3-epoch o2o protocol.
set -uo pipefail

YEAR=$1
shift
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
DATA=/root/autodl-tmp/data
STORE=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
O2O=$P/reports/dependencies/finals_pre/shared/o2o_labels.parquet
OUT=/root/e13_frozen_decay

if [ "$#" -eq 0 ]; then
  set -- 20260801 20260812 20260823
fi

mkdir -p "$OUT"
cd "$P"
for SEED in "$@"; do
  DIR=$OUT/y${YEAR}_seed${SEED}
  LOG=$OUT/y${YEAR}_seed${SEED}.log
  if [ -s "$DIR/oos_metrics.json" ]; then
    echo "[skip ] y${YEAR} seed${SEED}"
    continue
  fi
  echo "[start] y${YEAR} seed${SEED} $(date -Is)"
  if "$PY" scripts/evaluate_unified_microstructure.py \
      --data-root "$DATA" \
      --micro-store "$STORE" \
      --output-dir "$DIR" \
      --years "$YEAR" \
      --train-start-year 2019 \
      --training-mode expanding \
      --prediction-days 999 \
      --epochs 3 \
      --model-dim 96 \
      --kernels 3 15 60 \
      --tcn-blocks 3 \
      --tail-minutes 30 \
      --max-minutes 242 \
      --max-stocks 1200 \
      --min-train-days 700 \
      --learning-rate 4e-4 \
      --seed "$SEED" \
      --label-column ret_open_to_open \
      --extra-labels "$O2O" \
      --fast-pack >"$LOG" 2>&1; then
    echo "[done ] y${YEAR} seed${SEED} $(date -Is)"
  else
    echo "[FAIL ] y${YEAR} seed${SEED} $(date -Is)"
    tail -30 "$LOG"
    exit 1
  fi
done
echo "[E13 LANE DONE y${YEAR}] $(date -Is)"
