#!/usr/bin/env bash
# E3b: grouped cross-sectional context. Identical to the E3 full config except for
# --industry-context, and run on the same three seeds as E2c so the comparison is
# seed-matched rather than a single-run difference.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
OUT=$P/reports/dependencies/finals_pre/e3b_industry_context
mkdir -p "$OUT"
cd "$P"
export PYTHONPATH=$P/experiments
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
      --industry-context --fast-pack > "$OUT/seed${SEED}.log" 2>&1; then
    echo "[done ] seed$SEED $(date -Is)"
  else
    echo "[FAIL ] seed$SEED $(date -Is)"; tail -20 "$OUT/seed${SEED}.log"
  fi
done
echo "[LANE DONE $* ] $(date -Is)"
