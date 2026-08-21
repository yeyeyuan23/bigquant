#!/usr/bin/env bash
# E1 orchestrator: 3 concurrent pathway ablations + Linear-85 baseline + scoring,
# then hand the freed GPU slot to E4 worker-2 (blocks 16-25, claimed via placeholders).
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
exp=$project_root/experiments/finals_pre_20260819/e1_pathway_ablation
out=$project_root/reports/dependencies/finals_pre_20260819/e1_pathway_ablation
e4=$project_root/reports/dependencies/finals_pre_20260819/e4_walkforward
mkdir -p "$out"
cd "$project_root"

echo "[start] linear85 (CPU bg) $(date -Is)"
"$python_bin" "$exp/linear85_baseline.py" --micro-store "$micro_store" --data-root "$data_root" \
  --output-dir "$out/linear85" --workers 24 > "$out/linear85.log" 2>&1 &
lin_pid=$!

run_gpu() {
  name="$1"; shift
  echo "[start] $name $(date -Is)"
  if "$python_bin" "$exp/evaluate_ablation.py" \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$out/$name" \
      --years 2024 --train-start-year 2019 --prediction-days 999 --epochs 3 \
      --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      "$@" > "$out/$name.log" 2>&1; then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}
run_gpu seq_only     --disable-statistics-path &
p1=$!
run_gpu stats_only   --disable-sequence-path &
p2=$!
run_gpu no_deepsets  --disable-cross-section &
p3=$!
wait $p1 $p2 $p3
wait $lin_pid && echo "[done ] linear85 $(date -Is)" || echo "[FAIL ] linear85 $(date -Is)"

echo "[start] e1 scoring $(date -Is)"
"$python_bin" scripts/score_submission_j_stability.py \
  "$project_root/reports/dependencies/finals_pre_20260819/e0_holdout_2024_e3/unified_microstructure_full_oos.parquet" \
  "$out/seq_only/unified_microstructure_full_oos.parquet" \
  "$out/stats_only/unified_microstructure_full_oos.parquet" \
  "$out/no_deepsets/unified_microstructure_full_oos.parquet" \
  "$out/linear85/unified_microstructure_full_oos.parquet" \
  --years 2024 --data-dir "$data_root" --reports-dir "$project_root/reports" \
  --cache-dir "$project_root/reports/dependencies/finals_pre_20260819/score_cache" \
  --output "$out/e1_j_scores.json" --summary-csv "$out/e1_j_scores.csv" \
  > "$out/scoring_e1.log" 2>&1 && echo "[done ] e1 scoring" || echo "[FAIL ] e1 scoring"
echo "[E1 ALL DONE] $(date -Is)"

echo "[start] e4 worker-2 handoff $(date -Is)"
for i in $(seq 16 25); do
  bb=$(printf "block_%02d" "$i")
  mkdir -p "$e4/$bb"
  f="$e4/$bb/unified_microstructure_full_oos.parquet"
  if [ ! -f "$f" ]; then echo placeholder > "$f"; fi
done
for i in $(seq 16 25); do
  bb=$(printf "block_%02d" "$i")
  f="$e4/$bb/unified_microstructure_full_oos.parquet"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 10000 ]; then echo "[skip ] $bb"; continue; fi
  echo "[start] $bb (w2) $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$e4/$bb" \
      --years 2023 2024 --train-start-year 2019 --training-mode expanding --prediction-days 20 \
      --block-indices "$i" \
      --epochs 3 --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed 20260801 \
      > "$e4/$bb.w2.log" 2>&1; then echo "[done ] $bb (w2) $(date -Is)"; else echo "[FAIL ] $bb (w2) $(date -Is)"; fi
done
echo "[E4 W2 DONE] $(date -Is)"
