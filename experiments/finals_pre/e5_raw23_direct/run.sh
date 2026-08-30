#!/usr/bin/env bash
set -euo pipefail

root=/home/aiuser/work/e5_raw23_direct
python_bin=python
out=/home/aiuser/work/e5_raw23_results_c2c

"$python_bin" "$root/build_store.py"
"$python_bin" "$root/build_c2c_labels.py"
for seed in 20260801 20260812 20260823; do
  for arm in baseline17 raw40; do
    run_dir=$out/${arm}_seed${seed}
    if [[ -s "$run_dir/factor_2024.parquet" && -s "$run_dir/metrics.json" ]]; then
      echo "[skip] $arm seed=$seed"
      continue
    fi
    echo "[start] $arm seed=$seed $(date -Is)"
    "$python_bin" "$root/train.py" \
      --arm "$arm" --seed "$seed" --epochs 3 --output-dir "$run_dir"
    echo "[done] $arm seed=$seed $(date -Is)"
  done
done
"$python_bin" "$root/score.py"
