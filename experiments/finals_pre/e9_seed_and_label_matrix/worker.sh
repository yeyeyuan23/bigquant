#!/usr/bin/env bash
# One worker: claim the next unclaimed job under flock, run it, repeat.
# A shared queue instead of fixed lanes so no worker sits idle behind a long one.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
DATA=/root/autodl-tmp/data
STORE=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
O2O=$P/reports/dependencies/finals_pre/o2o_labels.parquet
OUT=$P/reports/dependencies/finals_pre/e9_seed_and_label_matrix
D=$P/experiments/finals_pre/e9_seed_and_label_matrix
ABL=$P/experiments/finals_pre/e3_pathway_ablation/evaluate_ablation.py
mkdir -p "$OUT/claims"; cd "$P"
COMMON="--data-root $DATA --micro-store $STORE --years 2024 --train-start-year 2019 \
--prediction-days 999 --epochs 3 --model-dim 96 --tcn-blocks 3 --tail-minutes 30 \
--max-minutes 242 --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --fast-pack"

while true; do
  job=$(flock "$OUT/claims/.lock" -c '
    while read -r line; do
      [ -z "$line" ] && continue
      name=$(echo "$line" | awk "{print \$2}")
      if [ ! -e "'"$OUT"'/claims/$name" ]; then
        touch "'"$OUT"'/claims/$name"; echo "$line"; exit 0
      fi
    done < "'"$D"'/jobs.txt"
    exit 1')
  [ -z "$job" ] && break
  set -- $job
  kind=$1; name=$2; seed=$3; lab=$4; shift 4
  dir=$OUT/$name
  if [ -s "$dir/oos_metrics.json" ]; then echo "[skip ] $name"; continue; fi
  extra=""
  [ "$lab" = "o2o" ] && extra="--label-column ret_open_to_open --extra-labels $O2O"
  echo "[start] $name $(date -Is)"
  if [ "$kind" = "abl" ]; then
    cmd=($PY "$ABL" $COMMON --kernels 3 15 60 --output-dir "$dir" --seed "$seed" "$@" $extra)
  else
    cmd=($PY scripts/evaluate_unified_microstructure.py $COMMON --kernels "$@" \
         --output-dir "$dir" --seed "$seed" $extra)
  fi
  if "${cmd[@]}" > "$OUT/$name.log" 2>&1; then
    echo "[done ] $name $(date -Is)"
  else
    echo "[FAIL ] $name $(date -Is)"; tail -5 "$OUT/$name.log"
  fi
done
echo "[WORKER EXIT] $(date -Is)"
