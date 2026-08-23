#!/usr/bin/env bash
# E0: the epoch curve. Produced by the mainline trainer itself with
# --eval-every-epoch, so mid-training and final scoring share one code path
# and no second implementation can drift.
set -uo pipefail
YEAR=$1
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
OUT=$P/reports/dependencies/finals_pre/e5_epoch_curve
cd "$P"
for SEED in 20260801 20260812 20260823; do
  DIR=$OUT/y${YEAR}_seed${SEED}
  if [ -s "$DIR/oos_metrics.json" ]; then echo "[skip ] y$YEAR seed$SEED"; continue; fi
  echo "[start] y$YEAR seed$SEED $(date -Is)"
  if "$PY" scripts/evaluate_unified_microstructure.py \
      --data-root /root/autodl-tmp/projects/bigquant/data \
      --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
      --output-dir "$DIR" --years "$YEAR" --train-start-year 2019 \
      --training-mode expanding --prediction-days 999 --epochs 6 \
      --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 \
      --max-minutes 242 --max-stocks 1200 --min-train-days 900 \
      --learning-rate 4e-4 --seed "$SEED" \
      --eval-every-epoch --fast-pack > "$OUT/y${YEAR}_seed${SEED}.log" 2>&1; then
    echo "[done ] y$YEAR seed$SEED $(date -Is)"
  else
    echo "[FAIL ] y$YEAR seed$SEED $(date -Is)"
  fi
done
echo "[LANE $YEAR DONE] $(date -Is)"
