#!/usr/bin/env bash
# E14: progressively simplify the k=15 architecture, three paired seeds.
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
trainer=$root/experiments/finals_pre/e3_pathway_ablation/evaluate_ablation.py
experiment=$root/experiments/finals_pre/e14_minimal_k15
output=$root/reports/dependencies/finals_pre/e14_minimal_k15
labels=$root/reports/dependencies/finals_pre/shared/o2o_labels.parquet
seeds=(20260801 20260812 20260823)

mkdir -p "$output/logs"
cd "$root"

run_one() {
  local arm=$1
  local seed=$2
  shift 2
  local outdir=$output/$arm/seed$seed
  local logfile=$output/logs/${arm}_seed${seed}.log
  if [[ -s $outdir/unified_microstructure_full_oos.parquet ]]; then
    echo "[skip ] $arm seed$seed $(date -Is)"
    return 0
  fi
  mkdir -p "$outdir"
  echo "[start] $arm seed$seed $(date -Is)"
  "$python_bin" "$trainer" \
    --data-root "$data_root" \
    --micro-store "$micro_store" \
    --output-dir "$outdir" \
    --years 2024 \
    --train-start-year 2019 \
    --prediction-days 999 \
    --epochs 3 \
    --model-dim 96 \
    --kernels 15 \
    --tcn-blocks 3 \
    --tail-minutes 30 \
    --max-minutes 242 \
    --max-stocks 1200 \
    --min-train-days 900 \
    --learning-rate 4e-4 \
    --seed "$seed" \
    --fast-pack \
    --label-column ret_open_to_open \
    --extra-labels "$labels" \
    "$@" >"$logfile" 2>&1
  echo "[done ] $arm seed$seed $(date -Is)"
}

run_arm() {
  local arm=$1
  shift
  local pids=()
  for seed in "${seeds[@]}"; do
    run_one "$arm" "$seed" "$@" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "[FAIL ] $arm; inspect $output/logs" >&2
    return 1
  fi
}

# Each arm removes exactly one more component than the previous arm.
run_arm k15_no_stats \
  --disable-statistics-path
run_arm k15_no_stats_no_fusion \
  --disable-statistics-path --disable-path-fusion
run_arm k15_no_stats_no_fusion_no_deepsets \
  --disable-statistics-path --disable-path-fusion --disable-cross-section
run_arm k15_minimal \
  --disable-statistics-path --disable-path-fusion --disable-cross-section --linear-head

candidate_args=()
for seed in "${seeds[@]}"; do
  baseline=$root/reports/dependencies/finals_pre/e9_seed_and_label_matrix/o2o_k15_$seed/unified_microstructure_full_oos.parquet
  candidate_args+=(--candidate "k15_baseline_$seed=$baseline")
  for arm in \
    k15_no_stats \
    k15_no_stats_no_fusion \
    k15_no_stats_no_fusion_no_deepsets \
    k15_minimal; do
    factor=$output/$arm/seed$seed/unified_microstructure_full_oos.parquet
    candidate_args+=(--candidate "${arm}_$seed=$factor")
  done
done

echo "[start] N residual RankIC $(date -Is)"
"$python_bin" "$root/experiments/finals_pre/e7_nscore/e7_layer1_ric.py" \
  --base "$root/reports/dependencies/finals_pre/e7_nscore/base/y_pool_oos.parquet" \
  --data-root "$data_root" \
  --years 2024 \
  --label-column ret_open_to_open \
  --extra-labels "$labels" \
  --residual isotonic \
  --output-dir "$output/nscore" \
  "${candidate_args[@]}" >"$output/logs/nscore.log" 2>&1

echo "[start] full-Barra scoring and paired tests $(date -Is)"
"$python_bin" "$experiment/score_simplification.py" \
  >"$output/logs/scoring.log" 2>&1
echo "[E14 DONE] $(date -Is)"
