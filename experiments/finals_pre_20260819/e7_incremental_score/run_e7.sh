#!/usr/bin/env bash
# CPU lane: N-score pipeline — base -> controls -> layer1 RIC -> layer2 paired delta.
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
pool_path=/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate454_store/features
pool_manifest=/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate454_store/candidate454_manifest.json
exp=$project_root/experiments/finals_pre_20260819/e7_incremental_score
out=$project_root/reports/dependencies/finals_pre_20260819/e7_incremental_score
mraw=$project_root/reports/dependencies/finals_pre_20260819/e0_holdout_2024_e3/unified_microstructure_full_oos.parquet
en454=$project_root/reports/dependencies/en454_baseline/baseline_partitions/year2024/candidate454_elasticnet_full_oos.parquet
mkdir -p "$out"
cd "$project_root"

if [ ! -s "$out/base/y_pool_oos.parquet" ]; then
  echo "[start] e7_base $(date -Is)"
  "$python_bin" "$exp/e7_build_base.py" --pool-path "$pool_path" --pool-manifest "$pool_manifest" \
    --data-root "$data_root" --output-dir "$out/base" --years 2024 --n-jobs 96 \
    > "$out/base.log" 2>&1 && echo "[done ] e7_base $(date -Is)" || { echo "[FAIL ] e7_base $(date -Is)"; exit 1; }
else
  echo "[skip ] e7_base"
fi

echo "[start] pool_factor_extract $(date -Is)"
"$python_bin" - <<PY > "$out/pool_factor_extract.log" 2>&1
import json, pandas as pd
manifest = json.load(open("$pool_manifest"))
cid = sorted(manifest["candidate_rows"])[0]
frame = pd.read_parquet("$pool_path", columns=["date","instrument",cid],
                        filters=[("date", ">=", pd.Timestamp("2024-01-01")), ("date", "<=", pd.Timestamp("2024-12-31"))])
frame = frame.rename(columns={cid: "value"})
frame.to_parquet("$out/pool_factor_control.parquet", index=False)
print("extracted", cid, len(frame))
PY
echo "[done ] pool_factor_extract"

echo "[start] e7_layer1 $(date -Is)"
"$python_bin" "$exp/e7_layer1_ric.py" --base "$out/base/y_pool_oos.parquet" --data-root "$data_root" \
  --years 2024 --output-dir "$out" \
  --candidate mraw="$mraw" \
  --candidate en454_control="$en454" \
  --candidate pool_factor_control="$out/pool_factor_control.parquet" \
  > "$out/layer1.log" 2>&1 && echo "[done ] e7_layer1 $(date -Is)" || echo "[FAIL ] e7_layer1 $(date -Is)"

echo "[start] e7_layer2 $(date -Is)"
"$python_bin" "$exp/e7_layer2_delta.py" --pool-path "$pool_path" --pool-manifest "$pool_manifest" \
  --data-root "$data_root" --years 2024 --train-start-year 2023 \
  --candidate mraw="$mraw" --n-jobs 48 --output-dir "$out" \
  > "$out/layer2.log" 2>&1 && echo "[done ] e7_layer2 $(date -Is)" || echo "[FAIL ] e7_layer2 $(date -Is)"
echo "[E7 ALL DONE] $(date -Is)"
