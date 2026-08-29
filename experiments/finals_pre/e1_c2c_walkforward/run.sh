#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
out=$root/reports/dependencies/finals_pre/e1_c2c_walkforward

mkdir -p "$out"
cd "$root"
"$python_bin" scripts/evaluate_unified_microstructure.py \
  --data-root /root/autodl-tmp/data \
  --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
  --output-dir "$out/model" \
  --years 2023 2024 --train-start-year 2019 \
  --training-mode expanding --prediction-days 20 \
  --epochs 3 --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 \
  --tail-minutes 30 --max-minutes 242 --max-stocks 1200 \
  --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
  --label-column ret_close_to_close --fast-pack --reuse-checkpoints
"$python_bin" experiments/finals_pre/e1_c2c_walkforward/score.py
