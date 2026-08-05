#!/usr/bin/env bash
set -euo pipefail

repo_root=/root/autodl-tmp/projects/bigquant-unified-integration
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
candidate_root=/root/autodl-tmp/candidate454_completion_full_2019_2024/candidate454_store
baseline_root="$repo_root/reports/dependencies/temporal_incremental_objective_20260803/candidate454_elasticnet_oos_2019_2024"
output_root="$repo_root/reports/x_residual_retrain_v1"

cd "$repo_root"
mkdir -p "$output_root/logs"
printf '%s\n' "$$" >"$output_root/master.pid"

tree_pids=()
for seed in 20260731 20260732 20260733; do
  seed_root="$output_root/tree_seed_${seed}"
  mkdir -p "$seed_root"
  OMP_NUM_THREADS=48 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  "$python_bin" scripts/evaluate_unified_tree_residual.py \
    --data-root "$data_root" \
    --output-dir "$seed_root" \
    --years 2023 2024 \
    --train-start-year 2019 \
    --candidate-pool "$candidate_root/features" \
    --candidate-manifest "$candidate_root/candidate454_manifest.json" \
    --expected-candidate-count 454 \
    --residual-baseline-route "$baseline_root/candidate454_elasticnet_full_oos.parquet" \
    --residual-baseline-manifest "$baseline_root/run_manifest.json" \
    --num-leaves 63 \
    --learning-rate 0.03 \
    --n-estimators 800 \
    --n-jobs 48 \
    --seed "$seed" \
    >"$output_root/logs/tree_seed_${seed}.log" 2>&1 &
  tree_pids+=("$!")
done
printf '%s\n' "${tree_pids[@]}" >"$output_root/tree_worker.pids"

for seed in 20260801 20260802 20260803; do
  seed_root="$output_root/mlp_seed_${seed}"
  mkdir -p "$seed_root"
  "$python_bin" scripts/evaluate_unified_mlp_residual.py \
    --data-root "$data_root" \
    --output-dir "$seed_root" \
    --years 2023 2024 \
    --train-start-year 2019 \
    --candidate-pool "$candidate_root/features" \
    --candidate-manifest "$candidate_root/candidate454_manifest.json" \
    --expected-candidate-count 454 \
    --residual-baseline-route "$baseline_root/candidate454_elasticnet_full_oos.parquet" \
    --residual-baseline-manifest "$baseline_root/run_manifest.json" \
    --hidden-dims 1024 512 256 \
    --dropout 0.12 \
    --epochs 12 \
    --train-stride 2 \
    --max-stocks 1200 \
    --learning-rate 0.0004 \
    --orthogonality-weight 0.05 \
    --seed "$seed" \
    >"$output_root/logs/mlp_seed_${seed}.log" 2>&1
done

for worker_pid in "${tree_pids[@]}"; do
  wait "$worker_pid"
done

touch "$output_root/TRAINING_COMPLETE"
