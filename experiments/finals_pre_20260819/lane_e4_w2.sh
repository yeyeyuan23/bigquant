#!/usr/bin/env bash
# E4 worker-2 standalone: blocks 16-25; placeholders (<=10KB) count as pending.
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
e4=$project_root/reports/dependencies/finals_pre_20260819/e4_walkforward
cd "$project_root"
for i in $(seq 16 25); do
  bb=$(printf "block_%02d" "$i")
  f="$e4/$bb/unified_microstructure_full_oos.parquet"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 10000 ]; then echo "[skip ] $bb (w2)"; continue; fi
  echo "[start] $bb (w2) $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$e4/$bb" \
      --years 2023 2024 --train-start-year 2019 --training-mode expanding --prediction-days 20 \
      --block-indices "$i" \
      --epochs 3 --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      > "$e4/$bb.w2.log" 2>&1; then echo "[done ] $bb (w2) $(date -Is)"; else echo "[FAIL ] $bb (w2) $(date -Is)"; fi
done
echo "[E4 W2 DONE] $(date -Is)"
