#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/projects/bigquant-candidate454-completion"
PYTHON_BIN="/root/autodl-tmp/conda-envs/quant/bin/python"
DATA_ROOT="/root/autodl-tmp/projects/bigquant/data"
WORK_ROOT="/root/autodl-tmp/candidate454_completion_full_2019_2024"
LOG_ROOT="$WORK_ROOT/logs"
MAX_POLLS="${CANDIDATE454_MAX_POLLS:-360}"

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

"$PYTHON_BIN" scripts/audit_candidate_source_provenance.py \
  --json "$WORK_ROOT/validation/candidate_source_provenance.json" \
  --csv "$WORK_ROOT/validation/candidate_source_provenance.csv" \
  >"$LOG_ROOT/candidate_source_provenance.log" 2>&1

wait_for_file() {
  local path="$1"
  local label="$2"
  local poll=0
  while [[ ! -s "$path" ]]; do
    poll=$((poll + 1))
    if (( poll > MAX_POLLS )); then
      echo "timed out waiting for $label: $path" >&2
      exit 1
    fi
    echo "[$(date '+%F %T')] waiting for $label ($poll/$MAX_POLLS)"
    sleep 60
  done
}

wait_for_file "$WORK_ROOT/remaining132/remaining132_report.json" "remaining132"
wait_for_file "$WORK_ROOT/gtja149/gtja149_report.json" "gtja149"
wait_for_file "$WORK_ROOT/haitong/haitong_raw_panel_2019_2024.parquet" "haitong panel"

"$PYTHON_BIN" scripts/factor_wiki_remaining/validate_prefix_invariance.py \
  --work "$WORK_ROOT" \
  --period-tag 2019_2024 \
  --cutoff 2023-06-30 \
  --output-dir "$WORK_ROOT/validation" \
  2>&1 | tee "$LOG_ROOT/prefix_remaining132.log"

"$PYTHON_BIN" scripts/build_direct6_feature_matrix.py \
  --data-root "$DATA_ROOT" \
  --haitong-panel "$WORK_ROOT/haitong/haitong_raw_panel_2019_2024.parquet" \
  --output-dir "$WORK_ROOT/direct6" \
  --years 2019 2020 2021 2022 2023 2024 \
  2>&1 | tee "$LOG_ROOT/direct6.log"

CUTOFF_ROOT="$WORK_ROOT/prefix_cutoff_2023_06_30"
"$PYTHON_BIN" scripts/build_gtja149_feature_matrix.py \
  --data-root "$DATA_ROOT" \
  --candidate-root "$REPO_ROOT/src/bigalpha2026/candidates" \
  --external-root "$REPO_ROOT/third_party/aurumq_gtja191" \
  --output-dir "$CUTOFF_ROOT/gtja149" \
  --years 2019 2020 2021 2022 2023 \
  --end-date 2023-06-30 \
  >"$LOG_ROOT/prefix_gtja149_rebuild.log" 2>&1

"$PYTHON_BIN" scripts/build_direct6_feature_matrix.py \
  --data-root "$DATA_ROOT" \
  --haitong-panel "$WORK_ROOT/haitong/haitong_raw_panel_2019_2024.parquet" \
  --output-dir "$CUTOFF_ROOT/direct6" \
  --years 2019 2020 2021 2022 2023 \
  --end-date 2023-06-30 \
  >"$LOG_ROOT/prefix_direct6_rebuild.log" 2>&1

"$PYTHON_BIN" scripts/validate_wide_prefix_invariance.py \
  --full "$WORK_ROOT/gtja149/gtja149_features_wide.parquet" \
  --rebuilt "$CUTOFF_ROOT/gtja149/gtja149_features_wide.parquet" \
  --candidate-ids "$WORK_ROOT/gtja149/gtja149_lineage.json" \
  --cutoff 2023-06-30 \
  --output "$WORK_ROOT/validation/gtja149_prefix_invariance.json" \
  2>&1 | tee "$LOG_ROOT/prefix_gtja149_compare.log"

"$PYTHON_BIN" scripts/validate_wide_prefix_invariance.py \
  --full "$WORK_ROOT/direct6/direct6_features_wide.parquet" \
  --rebuilt "$CUTOFF_ROOT/direct6/direct6_features_wide.parquet" \
  --candidate-ids "$WORK_ROOT/direct6/direct6_report.json" \
  --candidate-ids-format report \
  --cutoff 2023-06-30 \
  --output "$WORK_ROOT/validation/direct6_prefix_invariance.json" \
  2>&1 | tee "$LOG_ROOT/prefix_direct6_compare.log"

"$PYTHON_BIN" scripts/assemble_candidate454_feature_store.py \
  --data-root "$DATA_ROOT" \
  --provenance-csv "$WORK_ROOT/validation/candidate_source_provenance.csv" \
  --base-pool "$DATA_ROOT/runtime/all156_full_2019_2024_corrected/factors/candidate_pool.parquet" \
  --remaining-features "$WORK_ROOT/remaining132/remaining132_features_wide.parquet" \
  --remaining-availability "$WORK_ROOT/remaining132/remaining132_availability_wide.parquet" \
  --remaining-lineage "$WORK_ROOT/remaining132/remaining132_lineage.csv" \
  --gtja-features "$WORK_ROOT/gtja149/gtja149_features_wide.parquet" \
  --gtja-availability "$WORK_ROOT/gtja149/gtja149_availability_wide.parquet" \
  --gtja-lineage "$WORK_ROOT/gtja149/gtja149_lineage.json" \
  --cicc13 "$DATA_ROOT/runtime/factor_wiki_latent_20260731/candidate_pool_latent_cicc13_delta.parquet" \
  --pv16 "$DATA_ROOT/runtime/factor_wiki_latent_20260731/candidate_pool_latent_pv16_delta.parquet" \
  --direct-features "$WORK_ROOT/direct6/direct6_features_wide.parquet" \
  --direct-availability "$WORK_ROOT/direct6/direct6_availability_wide.parquet" \
  --direct-report "$WORK_ROOT/direct6/direct6_report.json" \
  --output-root "$WORK_ROOT/candidate454_store" \
  --expected-count 454 \
  --manifest-name candidate454_manifest.json \
  --years 2019 2020 2021 2022 2023 2024 \
  2>&1 | tee "$LOG_ROOT/assemble_candidate454.log"

"$PYTHON_BIN" scripts/validate_candidate454_store.py \
  --store "$WORK_ROOT/candidate454_store" \
  --expected-count 454 \
  --output "$WORK_ROOT/candidate454_store/candidate454_validation.json" \
  2>&1 | tee "$LOG_ROOT/validate_candidate454_store.log"

echo "[$(date '+%F %T')] candidate454 store complete: $WORK_ROOT/candidate454_store"
