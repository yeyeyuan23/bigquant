#!/usr/bin/env bash
# E2b: kernel-value sensitivity — same log-spaced design, different values.
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
  if [ -s "$outdir/unified_microstructure_full_oos.parquet" ]; then echo "[skip ] $name"; return 0; fi
  echo "[start] $name $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$outdir" \
      --years 2024 --train-start-year 2019 --training-mode expanding --prediction-days 999 \
      --epochs 3 --model-dim 96 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      "$@" > "$out_root/${name}.log" 2>&1; then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}
run_variant e2b_kernel_5_30_120 --kernels 5 30 120
run_variant e2b_kernel_2_10_45 --kernels 2 10 45
"$python_bin" scripts/score_submission_j_stability.py \
  "$out_root/e5_reference_e3/unified_microstructure_full_oos.parquet" \
  "$out_root/e2b_kernel_5_30_120/unified_microstructure_full_oos.parquet" \
  "$out_root/e2b_kernel_2_10_45/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$out_root/score_cache" \
  --output "$out_root/shared/e2b_j_scores.json" --summary-csv "$out_root/shared/e2b_j_scores.csv" \
  > "$out_root/scoring_e2b.log" 2>&1 && echo "[done ] scoring_e2b" || echo "[FAIL ] scoring_e2b"
echo "[E2B ALL DONE] $(date -Is)"
