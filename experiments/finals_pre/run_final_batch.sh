#!/usr/bin/env bash
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
D=/root/autodl-tmp/data
R=$P/reports/dependencies/finals_pre
cd "$P"
echo "[start] e1_channel_importance $(date -Is)"
"$PY" experiments/finals_pre/e1_channel_importance/perm_importance.py \
  --checkpoint $R/e5_reference_e3/unified_microstructure_block_00_checkpoint.pt \
  --micro-store /root/autodl-tmp/unified_microstructure_store_v2_2019_2024 \
  --data-root $D --year 2024 --day-stride 4 --output-dir $R/e1_channel_importance \
  > $R/e1_channel_importance/perm.log 2>&1 && echo "[done ] e1_channel_importance $(date -Is)" || echo "[FAIL ] e1_channel_importance $(date -Is)" &
PERM=$!
echo "[start] e5_build $(date -Is)"
"$PY" experiments/finals_pre/e8_en_blend/e5_blend.py \
  --m-series $R/e4_walkforward/merged_full_oos.parquet \
  --baseline $P/reports/dependencies/en454_baseline/candidate454_elasticnet_oos_2019_2024/candidate454_elasticnet_full_oos.parquet \
  --data-root $D --years 2023 2024 --output-dir $R/e8_en_blend \
  > $R/e8_en_blend/e5.log 2>&1 && echo "[done ] e5_build $(date -Is)" || { echo "[FAIL ] e5_build"; }
if [ -s $R/e8_en_blend/blend50.parquet ]; then
  echo "[start] e5_scoring $(date -Is)"
  "$PY" scripts/score_submission_j_stability.py \
    $R/e8_en_blend/baseline_en454.parquet $R/e8_en_blend/m_walkforward.parquet \
    $R/e8_en_blend/blend10.parquet $R/e8_en_blend/blend25.parquet $R/e8_en_blend/blend50.parquet \
    --years 2023 2024 --data-dir $D --reports-dir $P/reports \
    --cache-dir $R/score_cache \
    --output $R/e8_en_blend/e5_j_scores.json --summary-csv $R/e8_en_blend/e5_j_scores.csv \
    > $R/e8_en_blend/scoring_e5.log 2>&1 && echo "[done ] e5_scoring $(date -Is)" || echo "[FAIL ] e5_scoring $(date -Is)"
fi
wait $PERM
echo "[FINAL BATCH DONE] $(date -Is)"
