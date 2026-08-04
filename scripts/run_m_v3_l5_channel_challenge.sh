#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/projects/bigquant-m-v3-l5-channels-20260803
integration_root=/root/autodl-tmp/projects/bigquant-unified-integration
m_v2_root=/root/autodl-tmp/projects/bigquant-m-v2-l5-gated-20260803/reports/m_v2_flow_gated_20260803
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
deep_book_dir=/root/autodl-tmp/m_v3_deep_book_context_2019_2024
run_root="$project_root/reports/m_v3_l5_channels_20260803"

mkdir -p "$run_root"
cd "$project_root"

for seed in 20260801 20260802 20260803; do
  output_dir="$run_root/holdout_2024_seed_${seed}"
  prediction="$output_dir/unified_microstructure_v3_full_oos.parquet"
  manifest="$output_dir/run_manifest.json"
  metrics="$output_dir/oos_metrics.json"
  if [[ -s "$prediction" && -s "$manifest" && -s "$metrics" ]]; then
    echo "seed=${seed} status=complete action=reuse"
    continue
  fi
  mkdir -p "$output_dir"
  "$python_bin" scripts/evaluate_unified_microstructure_v2.py \
    --data-root "$data_root" \
    --micro-store "$micro_store" \
    --deep-book-dir "$deep_book_dir" \
    --min-deep-book-coverage 0.5 \
    --output-dir "$output_dir" \
    --years 2024 \
    --train-start-year 2019 \
    --training-mode expanding \
    --prediction-days 999 \
    --epochs 3 \
    --model-dim 96 \
    --kernels 3 15 60 \
    --tcn-blocks 3 \
    --tail-minutes 30 \
    --max-minutes 242 \
    --max-stocks 1200 \
    --min-train-days 900 \
    --learning-rate 4e-4 \
    --seed "$seed" \
    2>&1 | tee "$output_dir/train.log"
done

baseline="$integration_root/reports/m_expanding_history_retrain_20260803/strict_oos_route_2023_2024/unified_microstructure_strict_oos.parquet"
m_v2_ensemble="$run_root/m_dynamic_seed_ensemble.parquet"
m_v3_ensemble="$run_root/m_l5_seed_ensemble.parquet"

"$python_bin" scripts/ensemble_unified_routes.py \
  "$m_v2_root/holdout_2024_seed_20260801/unified_microstructure_v2_full_oos.parquet" \
  "$m_v2_root/holdout_2024_seed_20260802/unified_microstructure_v2_full_oos.parquet" \
  "$m_v2_root/holdout_2024_seed_20260803/unified_microstructure_v2_full_oos.parquet" \
  --output "$m_v2_ensemble"

"$python_bin" scripts/ensemble_unified_routes.py \
  "$run_root/holdout_2024_seed_20260801/unified_microstructure_v3_full_oos.parquet" \
  "$run_root/holdout_2024_seed_20260802/unified_microstructure_v3_full_oos.parquet" \
  "$run_root/holdout_2024_seed_20260803/unified_microstructure_v3_full_oos.parquet" \
  --output "$m_v3_ensemble"

score_args=(
  "$baseline"
  "$m_v2_root/holdout_2024_seed_20260801/unified_microstructure_v2_full_oos.parquet"
  "$m_v2_root/holdout_2024_seed_20260802/unified_microstructure_v2_full_oos.parquet"
  "$m_v2_root/holdout_2024_seed_20260803/unified_microstructure_v2_full_oos.parquet"
  "$run_root/holdout_2024_seed_20260801/unified_microstructure_v3_full_oos.parquet"
  "$run_root/holdout_2024_seed_20260802/unified_microstructure_v3_full_oos.parquet"
  "$run_root/holdout_2024_seed_20260803/unified_microstructure_v3_full_oos.parquet"
  "$m_v2_ensemble"
  "$m_v3_ensemble"
  --years 2024
  --data-dir "$data_root"
  --reports-dir "$integration_root/reports"
  --cache-dir "$run_root/score_cache"
  --output "$run_root/j_scores.json"
  --summary-csv "$run_root/j_scores.csv"
)
"$python_bin" scripts/score_submission_j_stability.py "${score_args[@]}"

"$python_bin" scripts/score_m_three_route_joint_diagnostic.py \
  --data-dir "$data_root" \
  --reports-dir "$integration_root/reports" \
  --year 2024 \
  --route old_m "$baseline" \
  --route m_dynamic_ensemble "$m_v2_ensemble" \
  --route m_l5_ensemble "$m_v3_ensemble" \
  --output "$run_root/three_route_joint_diagnostic.json" \
  --summary-csv "$run_root/three_route_joint_diagnostic.csv"

echo "status=complete run_root=$run_root"
