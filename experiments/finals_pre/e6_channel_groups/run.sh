#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
trainer=$root/experiments/finals_pre/e3_progressive_add/train_progressive.py
data=/root/autodl-tmp/data
store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
out=$root/reports/dependencies/finals_pre/e6_channel_groups/o2c/clock240

mkdir -p "$out"
cd "$root"

common=(
  --data-root "$data"
  --micro-store "$store"
  --years 2024
  --train-start-year 2019
  --prediction-days 999
  --epochs 3
  --model-dim 96
  --kernels 3 15 60
  --tcn-blocks 3
  --tail-minutes 30
  --max-minutes 240
  --max-stocks 1200
  --min-train-days 900
  --learning-rate 4e-4
  --label-column ret_next_open_to_close
  --fast-pack
)

run_arm() {
  local arm=$1
  local seed=$2
  shift 2
  local run_dir=$out/${arm}_seed${seed}
  local log=$out/${arm}_seed${seed}.log
  if [[ -s "$run_dir/unified_microstructure_full_oos.parquet" && -s "$run_dir/oos_metrics.json" ]]; then
    echo "[skip] $arm seed=$seed"
    return
  fi
  echo "[start] $arm seed=$seed $(date -Is)"
  "$python_bin" "$trainer" "${common[@]}" --output-dir "$run_dir" --seed "$seed" "$@" >"$log" 2>&1
  echo "[done] $arm seed=$seed $(date -Is)"
}

if (($#)); then
  seeds=("$@")
else
  seeds=(20260801 20260812 20260823)
fi

for seed in "${seeds[@]}"; do
  run_arm no_price_path "$seed" --drop-channel-group price
  run_arm no_book "$seed" --drop-channel-group book
  run_arm no_trading_structure "$seed" --drop-channel-group trade
  run_arm price_time_only "$seed" --drop-channel-group book trade
done

if [[ "${SKIP_SCORE:-0}" != 1 ]]; then
  "$python_bin" "$root/experiments/finals_pre/e6_channel_groups/score.py"
fi
echo "[all done] $(date -Is)"
