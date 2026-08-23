#!/usr/bin/env bash
# E7 addendum: true leave-one-out marginal contribution of one pool member.
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
pool_path=/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate454_store/features
pool_manifest=/root/autodl-tmp/candidate462_completion_full_2019_2024/candidate454_store/candidate454_manifest.json
exp=$project_root/experiments/finals_pre/e7_nscore
out=$project_root/reports/dependencies/finals_pre/e7_nscore
cd "$project_root"
cid=$("$python_bin" -c "import json; print(sorted(json.load(open(\"$pool_manifest\"))[\"candidate_rows\"])[0])")
echo "[loo ] excluded member: $cid"
echo "[start] loo_base $(date -Is)"
"$python_bin" "$exp/e7_build_base.py" --pool-path "$pool_path" --pool-manifest "$pool_manifest" \
  --data-root "$data_root" --output-dir "$out/base_loo" --years 2024 --n-jobs 96 \
  --exclude-candidate "$cid" \
  > "$out/base_loo.log" 2>&1 && echo "[done ] loo_base $(date -Is)" || { echo "[FAIL ] loo_base"; exit 1; }
echo "[start] loo_layer1 $(date -Is)"
"$python_bin" "$exp/e7_layer1_ric.py" --base "$out/base_loo/y_pool_oos.parquet" --data-root "$data_root" \
  --years 2024 --output-dir "$out/loo" \
  --candidate loo_member="$out/pool_factor_control.parquet" \
  > "$out/loo_layer1.log" 2>&1 && echo "[done ] loo_layer1 $(date -Is)" || echo "[FAIL ] loo_layer1"
echo "[E7 LOO DONE] $(date -Is)"
