#!/usr/bin/env bash
# E6b: train on the convention we are actually scored on.
# Everything matches the M_raw submission except --label-column, so the difference
# is attributable to the target definition and nothing else.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
OUT=$P/reports/dependencies/finals_pre/e6b_o2o_label
mkdir -p "$OUT"; cd "$P"
for SEED in "$@"; do
  DIR=$OUT/seed${SEED}
  if [ -s "$DIR/oos_metrics.json" ]; then echo "[skip ] seed$SEED"; continue; fi
  echo "[start] seed$SEED $(date -Is)"
  if "$PY" scripts/evaluate_unified_microstructure.py \
      --data-root /root/autodl-tmp/data \
      --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
      --output-dir "$DIR" --years 2024 --train-start-year 2019 \
      --training-mode expanding --prediction-days 999 --epochs 3 \
      --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 \
      --max-minutes 242 --max-stocks 1200 --min-train-days 900 \
      --learning-rate 4e-4 --seed "$SEED" \
      --label-column ret_open_to_open \
      --extra-labels "$P/reports/dependencies/finals_pre/o2o_labels.parquet" \
      --fast-pack > "$OUT/seed${SEED}.log" 2>&1; then
    echo "[done ] seed$SEED $(date -Is)"
  else
    echo "[FAIL ] seed$SEED $(date -Is)"; tail -20 "$OUT/seed${SEED}.log"
  fi
done
echo "[E6B LANE DONE $*] $(date -Is)"
