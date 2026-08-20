#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
candidate_root=/root/autodl-tmp/candidate454_completion_full_2019_2024/candidate454_store
suite_root="$repo_root/reports/dependencies/unified_alpha_fusion_suite_full_experts_v1"
stage_root="$suite_root/residual_boosting_end_to_end"
x_route="$suite_root/mlp_base/unified_mlp_full_oos.parquet"

cd "$repo_root"
for seed in 20260801 20260802 20260803; do
  output_root="$stage_root/t_residual_seed_${seed}"
  merged="$output_root/unified_temporal_residual_full_oos.parquet"
  orthogonal="$output_root/unified_temporal_residual_orthogonal_full_oos.parquet"
  if [[ -s "$orthogonal" ]]; then
    echo "reuse t_residual seed=$seed"
    continue
  fi
  common=(
    --data-root "$data_root"
    --train-start-year 2019
    --candidate-pool "$candidate_root/features"
    --candidate-manifest "$candidate_root/candidate454_manifest.json"
    --expected-candidate-count 454
    --epochs 8
    --train-stride 2
    --model-dim 384
    --transformer-layers 6
    --attention-heads 8
    --feedforward-dim 1024
    --kernels 3 5 15
    --dropout 0.1
    --learning-rate 5e-4
    --max-stocks 1200
    --seed "$seed"
    --residual-route X "$x_route" 1.0
  )
  if [[ ! -s "$output_root/year2023/unified_temporal_2023_h2_oos.parquet" ]]; then
    "$python_bin" scripts/evaluate_unified_temporal_residual.py \
      "${common[@]}" \
      --output-dir "$output_root/year2023" \
      --years 2023 \
      --halves h2
  fi
  if [[ ! -s "$output_root/year2024/unified_temporal_2024_full_oos.parquet" ]]; then
    "$python_bin" scripts/evaluate_unified_temporal_residual.py \
      "${common[@]}" \
      --output-dir "$output_root/year2024" \
      --years 2024 \
      --halves h1 h2
  fi
  if [[ ! -s "$merged" ]]; then
    "$python_bin" scripts/merge_strict_oos_routes.py \
      --input "$output_root/year2023/unified_temporal_2023_h2_oos.parquet" \
      --input "$output_root/year2024/unified_temporal_2024_full_oos.parquet" \
      --output "$merged"
  fi
  "$python_bin" scripts/orthogonalize_oos_route.py \
    --candidate "$merged" \
    --base X "$x_route" \
    --output "$orthogonal" \
    --report "$output_root/orthogonalization_report.json"
  echo "done t_residual seed=$seed"
done
echo "all_temporal_residual_seeds_complete"
