#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/projects/bigquant-all156-temporal"
DATA_ROOT="/root/autodl-tmp/projects/bigquant/data"
CANDIDATE_STORE="/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate462_store"
CANDIDATE_POOL="${ALL618_CANDIDATE_POOL:-$CANDIDATE_STORE/features}"
CANDIDATE_MANIFEST="${ALL618_CANDIDATE_MANIFEST:-$CANDIDATE_STORE/candidate462_manifest.json}"
PYTHON_BIN="/root/autodl-tmp/conda-envs/quant/bin/python"
RUN_ID="${ALL618_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$PROJECT_ROOT/reports/all618_fusion_suite_$RUN_ID"
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
    "$PYTHON_BIN" scripts/evaluate_all156_temporal_bar1m.py \
      --data-root "$DATA_ROOT" \
      --output-dir "$work_dir" \
      --years "$year" \
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
      2>&1 | tee "$LOG_ROOT/${tag}_${year}_h2.log"
    "$PYTHON_BIN" scripts/evaluate_all156_temporal_h1.py \
      --data-root "$DATA_ROOT" \
      --work-dir "$work_dir" \
      --year "$year" \
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
      2>&1 | tee "$LOG_ROOT/${tag}_${year}_h1.log"
    route_parts+=("$work_dir/all618_fusion_${year}_full_oos.parquet")
  done
  "$PYTHON_BIN" scripts/combine_oos_routes.py "${route_parts[@]}" \
    --output "$RUN_ROOT/${tag}_full_oos.parquet"
}

# Two genuinely different temporal capacities; both use expanding 2019-history.
run_temporal fusion_base 256 4 768 6 2 1024
run_temporal fusion_deep 384 6 1024 8 2 1200

"$PYTHON_BIN" scripts/evaluate_all618_mlp.py \
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

"$PYTHON_BIN" scripts/evaluate_all618_mlp.py \
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
"$PYTHON_BIN" scripts/evaluate_all156_tree.py \
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

FUSION_BASE="$RUN_ROOT/fusion_base_full_oos.parquet"
FUSION_DEEP="$RUN_ROOT/fusion_deep_full_oos.parquet"
MLP_BASE="$RUN_ROOT/mlp_base/all618_mlp_full_oos.parquet"
MLP_WIDE="$RUN_ROOT/mlp_wide/all618_mlp_full_oos.parquet"
TREE_ROUTE="$RUN_ROOT/lightgbm/all618_lightgbm_full_oos.parquet"
ENSEMBLE_ROUTE="$RUN_ROOT/all618_equal_weight_ensemble.parquet"

"$PYTHON_BIN" scripts/ensemble_all156_routes.py \
  "$FUSION_BASE" "$FUSION_DEEP" "$MLP_BASE" "$MLP_WIDE" "$TREE_ROUTE" \
  --output "$ENSEMBLE_ROUTE"

"$PYTHON_BIN" scripts/score_submission_j_stability.py \
  "$FUSION_BASE" "$FUSION_DEEP" "$MLP_BASE" "$MLP_WIDE" "$TREE_ROUTE" "$ENSEMBLE_ROUTE" \
  --years 2023 2024 \
  --data-dir "$DATA_ROOT" \
  --reports-dir "$J_REPORT_ROOT" \
  --cache-dir "$RUN_ROOT/j_cache" \
  --output "$RUN_ROOT/j_stability.json" \
  --summary-csv "$RUN_ROOT/j_stability.csv" \
  2>&1 | tee "$LOG_ROOT/j_stability.log"

for baseline in "$FUSION_BASE" "$FUSION_DEEP" "$MLP_BASE" "$MLP_WIDE"; do
  baseline_name="$(basename "$baseline" .parquet)"
  "$PYTHON_BIN" scripts/score_all156_tree_increment.py \
    --baseline "$baseline" \
    --tree "$TREE_ROUTE" \
    --years 2023 2024 \
    --tree-weights 0.10 0.25 0.50 \
    --data-dir "$DATA_ROOT" \
    --reports-dir "$J_REPORT_ROOT" \
    --output "$RUN_ROOT/tree_increment_${baseline_name}.json" \
    2>&1 | tee "$LOG_ROOT/tree_increment_${baseline_name}.log"
done

ln -sfn "$RUN_ROOT" "$PROJECT_ROOT/reports/all618_fusion_suite_latest"
echo "[$(date '+%F %T')] All618 suite complete: $RUN_ROOT"
