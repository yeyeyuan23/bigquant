#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/projects/bigquant-all156-temporal"
DATA_ROOT="/root/autodl-tmp/projects/bigquant/data"
CANDIDATE_STORE="/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate462_store"
CANDIDATE_POOL="${UNIFIED_CANDIDATE_POOL:-$CANDIDATE_STORE/features}"
CANDIDATE_MANIFEST="${UNIFIED_CANDIDATE_MANIFEST:-$CANDIDATE_STORE/candidate462_manifest.json}"
PYTHON_BIN="/root/autodl-tmp/conda-envs/quant/bin/python"
RUN_ID="${UNIFIED_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_ROOT/reports/unified_alpha_fusion_suite_$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
J_REPORT_ROOT="/root/autodl-tmp/projects/bigquant/reports"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/preflight_candidate462_artifact.py \
  --pool "$CANDIDATE_POOL" \
  --manifest "$CANDIDATE_MANIFEST" \
  --expected-count 462 \
  --required-start 2019-01-02 \
  --required-end 2024-12-31 \
  --output "$RUN_ROOT/candidate462_preflight.json" \
  2>&1 | tee "$LOG_ROOT/preflight.log"

"$PYTHON_BIN" scripts/evaluate_unified_elasticnet.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_ROOT/elasticnet_baseline" \
  --years 2023 2024 \
  --train-start-year 2019 \
  --candidate-pool "$CANDIDATE_POOL" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --expected-candidate-count 462 \
  --train-days 60 \
  --prediction-days 20 \
  --alpha 0.001 \
  --l1-ratio 0.5 \
  2>&1 | tee "$LOG_ROOT/elasticnet_baseline.log"

run_temporal() {
  local tag="$1"
  local model_dim="$2"
  local layers="$3"
  local feedforward_dim="$4"
  local epochs="$5"
  local stride="$6"
  local max_stocks="$7"
  local route_parts=()
  local year
  for year in 2023 2024; do
    local work_dir="$RUN_ROOT/${tag}_${year}"
    mkdir -p "$work_dir"
    "$PYTHON_BIN" scripts/evaluate_unified_temporal.py \
      --data-root "$DATA_ROOT" \
      --output-dir "$work_dir" \
      --years "$year" \
      --halves h1 h2 \
      --train-start-year 2019 \
      --candidate-pool "$CANDIDATE_POOL" \
      --candidate-manifest "$CANDIDATE_MANIFEST" \
      --expected-candidate-count 462 \
      --model-dim "$model_dim" \
      --transformer-layers "$layers" \
      --attention-heads 8 \
      --feedforward-dim "$feedforward_dim" \
      --epochs "$epochs" \
      --train-stride "$stride" \
      --max-stocks "$max_stocks" \
      2>&1 | tee "$LOG_ROOT/${tag}_${year}.log"
    route_parts+=("$work_dir/unified_temporal_${year}_full_oos.parquet")
  done
  "$PYTHON_BIN" scripts/combine_oos_routes.py "${route_parts[@]}" \
    --output "$RUN_ROOT/${tag}_full_oos.parquet"
}

# Two genuinely different temporal capacities over Candidate462 histories.
run_temporal temporal_base 256 4 768 6 2 1024
run_temporal temporal_deep 384 6 1024 8 2 1200

"$PYTHON_BIN" scripts/evaluate_unified_mlp.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_ROOT/mlp_base" \
  --years 2023 2024 \
  --train-start-year 2019 \
  --candidate-pool "$CANDIDATE_POOL" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --expected-candidate-count 462 \
  --hidden-dims 1024 512 256 \
  --epochs 12 \
  --train-stride 2 \
  --max-stocks 1200 \
  2>&1 | tee "$LOG_ROOT/mlp_base.log"

"$PYTHON_BIN" scripts/evaluate_unified_mlp.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_ROOT/mlp_wide" \
  --years 2023 2024 \
  --train-start-year 2019 \
  --candidate-pool "$CANDIDATE_POOL" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --expected-candidate-count 462 \
  --hidden-dims 1536 768 384 \
  --epochs 15 \
  --train-stride 2 \
  --max-stocks 1400 \
  2>&1 | tee "$LOG_ROOT/mlp_wide.log"

# The tree route is mandatory and is always included in both standalone J and delta-J.
"$PYTHON_BIN" scripts/evaluate_unified_tree.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_ROOT/lightgbm" \
  --years 2023 2024 \
  --train-start-year 2019 \
  --candidate-pool "$CANDIDATE_POOL" \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --expected-candidate-count 462 \
  --num-leaves 63 \
  --n-estimators 800 \
  2>&1 | tee "$LOG_ROOT/lightgbm.log"

ELASTICNET_BASELINE="$RUN_ROOT/elasticnet_baseline/candidate462_elasticnet_full_oos.parquet"
TEMPORAL_BASE="$RUN_ROOT/temporal_base_full_oos.parquet"
TEMPORAL_DEEP="$RUN_ROOT/temporal_deep_full_oos.parquet"
MLP_BASE="$RUN_ROOT/mlp_base/unified_mlp_full_oos.parquet"
MLP_WIDE="$RUN_ROOT/mlp_wide/unified_mlp_full_oos.parquet"
TREE_ROUTE="$RUN_ROOT/lightgbm/unified_lightgbm_full_oos.parquet"

"$PYTHON_BIN" scripts/score_submission_j_stability.py \
  "$ELASTICNET_BASELINE" "$TEMPORAL_BASE" "$TEMPORAL_DEEP" \
  "$MLP_BASE" "$MLP_WIDE" "$TREE_ROUTE" \
  --years 2023 2024 \
  --data-dir "$DATA_ROOT" \
  --reports-dir "$J_REPORT_ROOT" \
  --cache-dir "$RUN_ROOT/j_cache" \
  --output "$RUN_ROOT/j_stability.json" \
  --summary-csv "$RUN_ROOT/j_stability.csv" \
  2>&1 | tee "$LOG_ROOT/j_stability.log"

for expert in "$TEMPORAL_BASE" "$TEMPORAL_DEEP" "$MLP_BASE" "$MLP_WIDE" "$TREE_ROUTE"; do
  expert_name="$(basename "$expert" .parquet)"
  "$PYTHON_BIN" scripts/score_unified_expert_increment.py \
    --baseline "$ELASTICNET_BASELINE" \
    --expert "$expert" \
    --expert-name "$expert_name" \
    --years 2023 2024 \
    --expert-weights 0.10 0.25 0.50 \
    --data-dir "$DATA_ROOT" \
    --reports-dir "$J_REPORT_ROOT" \
    --output "$RUN_ROOT/expert_increment_${expert_name}.json" \
    2>&1 | tee "$LOG_ROOT/expert_increment_${expert_name}.log"
done

ln -sfn "$RUN_ROOT" "$PROJECT_ROOT/reports/unified_alpha_fusion_suite_latest"
echo "[$(date '+%F %T')] Unified alpha fusion suite complete: $RUN_ROOT"
