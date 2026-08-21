#!/usr/bin/env bash
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
D=/root/autodl-tmp/projects/bigquant/data
R=$P/reports/dependencies/finals_pre_20260819
cd "$P"
echo "[start] perm_importance $(date -Is)"
"$PY" experiments/finals_pre_20260819/perm_importance/perm_importance.py \
  --checkpoint $R/e0_holdout_2024_e3/unified_microstructure_block_00_checkpoint.pt \
  --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
  --data-root $D --year 2024 --day-stride 4 --output-dir $R/perm_importance \
  > $R/perm_importance/perm.log 2>&1 && echo "[done ] perm_importance $(date -Is)" || echo "[FAIL ] perm_importance $(date -Is)" &
PERM=$!
echo "[start] e5_build $(date -Is)"
"$PY" experiments/finals_pre_20260819/e5_blend/e5_blend.py \
  --m-series $R/e4_walkforward/merged_full_oos.parquet \
  --baseline $P/reports/dependencies/en454_baseline/candidate454_elasticnet_oos_2019_2024/candidate454_elasticnet_full_oos.parquet \
  --data-root $D --years 2023 2024 --output-dir $R/e5_blend \
  > $R/e5_blend/e5.log 2>&1 && echo "[done ] e5_build $(date -Is)" || { echo "[FAIL ] e5_build"; }
if [ -s $R/e5_blend/blend50.parquet ]; then
  echo "[start] e5_scoring $(date -Is)"
  "$PY" scripts/score_submission_j_stability.py \
    $R/e5_blend/baseline_en454.parquet $R/e5_blend/m_walkforward.parquet \
    $R/e5_blend/blend10.parquet $R/e5_blend/blend25.parquet $R/e5_blend/blend50.parquet \
    --years 2023 2024 --data-dir $D --reports-dir $P/reports \
    --cache-dir $R/score_cache \
    --output $R/e5_blend/e5_j_scores.json --summary-csv $R/e5_blend/e5_j_scores.csv \
    > $R/e5_blend/scoring_e5.log 2>&1 && echo "[done ] e5_scoring $(date -Is)" || echo "[FAIL ] e5_scoring $(date -Is)"
fi
wait $PERM
echo "[FINAL BATCH DONE] $(date -Is)"
