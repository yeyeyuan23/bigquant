#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
trainer=$root/experiments/finals_pre/e3_progressive_add/train_progressive.py
data=/root/autodl-tmp/data
store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
out=$root/reports/dependencies/finals_pre/e3_progressive_add/c2c

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
  --max-minutes 242
  --max-stocks 1200
  --min-train-days 900
  --learning-rate 4e-4
  --label-column ret_close_to_close
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

for seed in 20260801 20260812 20260823; do
  run_arm p0_statistics_head "$seed" \
    --disable-sequence-path --disable-path-fusion --disable-cross-section
  run_arm p1_add_deepsets "$seed" \
    --disable-sequence-path --disable-path-fusion
  run_arm p2_add_tcn_last "$seed" --sequence-summary last
  run_arm p3_add_full_summaries "$seed"
done

"$python_bin" "$root/experiments/finals_pre/e3_progressive_add/score.py"
echo "[all done] $(date -Is)"
