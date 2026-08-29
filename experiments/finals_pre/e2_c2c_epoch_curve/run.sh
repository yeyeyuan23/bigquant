#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
out=$root/reports/dependencies/finals_pre/e2_c2c_epoch_curve

mkdir -p "$out"
cd "$root"
for seed in 20260801 20260812 20260823; do
  run_dir=$out/y2024_seed${seed}
  if [[ -s "$run_dir/unified_microstructure_block_00_checkpoint_epoch_06_oos.parquet" ]]; then
    echo "[skip] seed=$seed"
    continue
  fi
  "$python_bin" scripts/evaluate_unified_microstructure.py \
    --data-root /root/autodl-tmp/data \
    --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
    --output-dir "$run_dir" --years 2024 --train-start-year 2019 \
    --training-mode expanding --prediction-days 999 --epochs 6 \
    --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 \
    --max-minutes 242 --max-stocks 1200 --min-train-days 900 \
    --learning-rate 4e-4 --seed "$seed" --label-column ret_close_to_close \
    --eval-every-epoch --fast-pack
done
"$python_bin" experiments/finals_pre/e2_c2c_epoch_curve/score.py
