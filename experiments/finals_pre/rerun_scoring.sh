#!/usr/bin/env bash
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
out_root=$project_root/reports/dependencies/finals_pre
cd "$project_root"
"$python_bin" scripts/score_submission_j_stability.py \
  "$out_root/e5_reference_e3/unified_microstructure_full_oos.parquet" \
  "$out_root/e5_reference_e6/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$out_root/score_cache" \
  --output "$out_root/holdout_j_scores.json" --summary-csv "$out_root/holdout_j_scores.csv" \
  > "$out_root/scoring_e0.log" 2>&1 && echo "[done ] scoring_e0 $(date -Is)" || echo "[FAIL ] scoring_e0 $(date -Is)"
"$python_bin" scripts/score_submission_j_stability.py \
  "$out_root/e5_reference_e3/unified_microstructure_full_oos.parquet" \
  "$out_root/e5_reference_e6/unified_microstructure_full_oos.parquet" \
  "$out_root/e2a_kernel_k3/unified_microstructure_full_oos.parquet" \
  "$out_root/e2a_kernel_k15/unified_microstructure_full_oos.parquet" \
  "$out_root/e2a_kernel_k60/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$out_root/score_cache" \
  --output "$out_root/all_variants_j_scores.json" --summary-csv "$out_root/all_variants_j_scores.csv" \
  > "$out_root/scoring_all.log" 2>&1 && echo "[done ] scoring_all $(date -Is)" || echo "[FAIL ] scoring_all $(date -Is)"
echo "[SCORING DONE] $(date -Is)"
