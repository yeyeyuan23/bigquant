#!/usr/bin/env bash
# GPU lane 2: E4 multi-block walk-forward, blocks 0..25 over 2023-2024, one dir per block (resumable).
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
out_root=$project_root/reports/dependencies/finals_pre_20260819/e4_walkforward
mkdir -p "$out_root"
cd "$project_root"
for i in $(seq 0 25); do
  bb=$(printf "block_%02d" "$i")
  outdir="$out_root/$bb"
  if [ -s "$outdir/unified_microstructure_full_oos.parquet" ]; then echo "[skip ] $bb"; continue; fi
  echo "[start] $bb $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$outdir" \
      --years 2023 2024 --train-start-year 2019 --training-mode expanding --prediction-days 20 \
      --block-indices "$i" \
      --epochs 3 --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      > "$out_root/$bb.log" 2>&1; then echo "[done ] $bb $(date -Is)"; else echo "[FAIL ] $bb $(date -Is)"; fi
done
echo "[E4 LANE DONE] $(date -Is)"
