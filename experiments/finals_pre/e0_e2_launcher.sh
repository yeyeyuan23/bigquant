#!/usr/bin/env bash
# E0 (holdout rebuild e3/e6) + E2 (single-scale kernels) — sequential, single GPU.
# Protocol: expanding 2019-2023 train -> 2024 untouched predict, seed 20260801.
set -uo pipefail

project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
out_root=$project_root/reports/dependencies/finals_pre

cd "$project_root"

run_variant() {
  name="$1"; shift
  outdir="$out_root/$name"
  if [ -s "$outdir/unified_microstructure_full_oos.parquet" ]; then
    echo "[skip ] $name already complete"
    return 0
  fi
  echo "[start] $name $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$outdir" \
      --years 2024 --train-start-year 2019 --training-mode expanding --prediction-days 999 \
      --model-dim 96 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      "$@" > "$out_root/${name}.log" 2>&1; then
    echo "[done ] $name $(date -Is)"
  else
    echo "[FAIL ] $name $(date -Is) — see ${name}.log"
  fi
}

run_variant e0_holdout_2024_e3 --epochs 3 --kernels 3 15 60
run_variant e0_holdout_2024_e6 --epochs 6 --kernels 3 15 60
run_variant e2_kernel_k3  --epochs 3 --kernels 3
run_variant e2_kernel_k15 --epochs 3 --kernels 15
run_variant e2_kernel_k60 --epochs 3 --kernels 60

echo "[train phase complete] $(date -Is)"

# Best-effort scoring: reproduce the original e3-vs-e6 holdout selection evidence.
"$python_bin" scripts/score_submission_j_stability.py \
  "$out_root/e0_holdout_2024_e3/unified_microstructure_full_oos.parquet" \
  "$out_root/e0_holdout_2024_e6/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$out_root/score_cache" \
  --output "$out_root/holdout_j_scores.json" \
  --summary-csv "$out_root/holdout_j_scores.csv" \
  > "$out_root/scoring_e0.log" 2>&1 && echo "[done ] scoring_e0" || echo "[FAIL ] scoring_e0 — see scoring_e0.log"

# Extended scoring across all five variants (kernel comparison table).
"$python_bin" scripts/score_submission_j_stability.py \
  "$out_root/e0_holdout_2024_e3/unified_microstructure_full_oos.parquet" \
  "$out_root/e0_holdout_2024_e6/unified_microstructure_full_oos.parquet" \
  "$out_root/e2_kernel_k3/unified_microstructure_full_oos.parquet" \
  "$out_root/e2_kernel_k15/unified_microstructure_full_oos.parquet" \
  "$out_root/e2_kernel_k60/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$out_root/score_cache" \
  --output "$out_root/all_variants_j_scores.json" \
  --summary-csv "$out_root/all_variants_j_scores.csv" \
  > "$out_root/scoring_all.log" 2>&1 && echo "[done ] scoring_all" || echo "[FAIL ] scoring_all — see scoring_all.log"

echo "[ALL DONE] $(date -Is)"
