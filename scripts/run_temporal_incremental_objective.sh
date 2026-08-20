#!/usr/bin/env bash
set -euo pipefail

variant=${1:-residual_orth_stable}
repo_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
candidate_root=/root/autodl-tmp/candidate454_completion_full_2019_2024/candidate454_store
run_root="$repo_root/reports/dependencies/temporal_incremental_objective_20260803"
baseline_root="$run_root/candidate454_elasticnet_oos_2019_2024"
baseline_route="$baseline_root/candidate454_elasticnet_full_oos.parquet"
baseline_manifest="$baseline_root/run_manifest.json"
output_root="$run_root/$variant/seed_20260803"

case "$variant" in
  residual)
    orthogonality_weight=0.0
    stability_weight=0.0
    ;;
  residual_orth)
    orthogonality_weight=0.10
    stability_weight=0.0
    ;;
  residual_orth_stable)
    orthogonality_weight=0.10
    stability_weight=0.30
    ;;
  *)
    echo "unknown variant: $variant" >&2
    echo "expected residual, residual_orth, or residual_orth_stable" >&2
    exit 2
    ;;
esac

cd "$repo_root"
mkdir -p "$baseline_root" "$output_root"

if [[ ! -s "$baseline_route" || ! -s "$baseline_manifest" ]]; then
  partition_dirs=()
  partition_pids=()
  for year in 2019 2020 2021 2022 2023 2024; do
    partition="$run_root/baseline_partitions/year${year}"
    partition_dirs+=("$partition")
    if [[ -s "$partition/candidate454_elasticnet_full_oos.parquet" \
      && -s "$partition/run_manifest.json" ]]; then
      continue
    fi
    mkdir -p "$partition"
    first_year_mode=()
    if [[ "$year" == 2019 ]]; then
      first_year_mode=(--continuous-oos)
      partition_train_start=2019
    else
      first_year_mode=(--preserve-missing-oos)
      partition_train_start=$((year - 1))
    fi
    (
      OMP_NUM_THREADS="${BASELINE_THREADS:-4}" \
      OPENBLAS_NUM_THREADS="${BASELINE_THREADS:-4}" \
      MKL_NUM_THREADS="${BASELINE_THREADS:-4}" \
      "$python_bin" scripts/evaluate_unified_elasticnet.py \
        --data-root "$data_root" \
        --output-dir "$partition" \
        --years "$year" \
        --train-start-year "$partition_train_start" \
        --candidate-pool "$candidate_root/features" \
        --candidate-manifest "$candidate_root/candidate454_manifest.json" \
        --expected-candidate-count 454 \
        --train-days 60 \
        --prediction-days 20 \
        --alpha 0.001 \
        --l1-ratio 0.5 \
        --max-iter 20000 \
        "${first_year_mode[@]}"
    ) >"$partition/run.log" 2>&1 &
    partition_pids+=("$!")
  done
  for pid in "${partition_pids[@]}"; do
    wait "$pid"
  done
  "$python_bin" scripts/merge_candidate454_elasticnet_oos.py \
    "${partition_dirs[@]}" \
    --output-dir "$baseline_root"
fi

"$python_bin" scripts/evaluate_unified_temporal.py \
  --data-root "$data_root" \
  --output-dir "$output_root" \
  --years 2023 \
  --halves h1 h2 \
  --train-start-year 2019 \
  --candidate-pool "$candidate_root/features" \
  --candidate-manifest "$candidate_root/candidate454_manifest.json" \
  --expected-candidate-count 454 \
  --epochs "${EPOCHS:-8}" \
  --train-stride "${TRAIN_STRIDE:-1}" \
  --model-dim 128 \
  --transformer-layers 2 \
  --attention-heads 8 \
  --feedforward-dim 384 \
  --kernels 3 5 15 \
  --history-mode dense \
  --dropout 0.1 \
  --learning-rate 0.0005 \
  --max-stocks "${MAX_STOCKS:-1200}" \
  --seed 20260803 \
  --target-mode candidate454_residual \
  --residual-baseline-route "$baseline_route" \
  --residual-baseline-manifest "$baseline_manifest" \
  --residual-orthogonality-weight "$orthogonality_weight" \
  --residual-stability-weight "$stability_weight" \
  --residual-correlation-floor 0.0

route="$output_root/unified_temporal_2023_full_oos.parquet"
"$python_bin" scripts/score_route_periods.py \
  "$route" \
  --years 2023 \
  --data-dir "$data_root" \
  --reports-dir reports \
  --output "$output_root/j_half_year_2023.json"

"$python_bin" scripts/score_submission_j_stability.py \
  "$route" \
  --years 2023 \
  --data-dir "$data_root" \
  --reports-dir reports \
  --output "$output_root/j_single_route_2023.json" \
  --summary-csv "$output_root/j_single_route_2023.csv"
