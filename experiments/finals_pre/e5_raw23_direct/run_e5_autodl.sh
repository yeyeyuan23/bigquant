#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/projects/bigquant-default
PYTHON=/root/autodl-tmp/conda-envs/quant/bin/python
CODE=$ROOT/experiments/finals_pre/e5_raw23_direct
STORE=/root/bigquant_private_data/e5_raw40_2023_2024_parquet
OUT=$ROOT/reports/dependencies/finals_pre/e5_raw23_direct/c2c_autodl

if [[ ! -s "$STORE/export_manifest.json" ]]; then
  echo "missing audited Parquet store manifest: $STORE/export_manifest.json" >&2
  exit 1
fi
"$PYTHON" "$CODE/build_c2c_labels.py"
if [[ ! -s "$CODE/c2c_labels.parquet" ]]; then
  echo "missing C2C labels: $CODE/c2c_labels.parquet" >&2
  exit 1
fi

mkdir -p "$OUT"
for seed in 20260801 20260812 20260823; do
  for arm in baseline17 raw40; do
    run_dir=$OUT/${arm}_seed${seed}
    log=$OUT/${arm}_seed${seed}.log
    if [[ -s "$run_dir/factor_2024.parquet" && -s "$run_dir/checkpoint.pt" ]]; then
      echo "[skip] arm=$arm seed=$seed"
      continue
    fi
    echo "[start] arm=$arm seed=$seed $(date --iso-8601=seconds)"
    "$PYTHON" -u "$CODE/e5_train_autodl.py" \
      --arm "$arm" \
      --seed "$seed" \
      --epochs 3 \
      --threads 8 \
      --device cuda \
      --store "$STORE" \
      --labels "$CODE/c2c_labels.parquet" \
      --output-dir "$run_dir" \
      >"$log" 2>&1
    echo "[done] arm=$arm seed=$seed $(date --iso-8601=seconds)"
  done
done
touch "$OUT/training_all_done"
echo "[all done] $(date --iso-8601=seconds)"
