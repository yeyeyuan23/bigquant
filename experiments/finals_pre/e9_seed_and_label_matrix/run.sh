#!/usr/bin/env bash
# E9: bring every ablation to 3 seeds, and redo the E3 family + the contested
# kernel group under matched o2o training.
#
# Three questions this closes:
#   P1  does the E3 stability ladder survive when TRAINING also uses o2o?
#   P2  is (2,10,45) really better, under matched training and enough seeds?
#   P3  every single-seed number in the deck becomes a 3-seed mean +/- scatter.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
DATA=/root/autodl-tmp/data
STORE=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
O2O=$P/reports/dependencies/finals_pre/shared/o2o_labels.parquet
OUT=$P/reports/dependencies/finals_pre/e9_seed_and_label_matrix
ABL=$P/experiments/finals_pre/e3_pathway_ablation/evaluate_ablation.py
mkdir -p "$OUT"; cd "$P"

COMMON="--data-root $DATA --micro-store $STORE --years 2024 --train-start-year 2019 \
--prediction-days 999 --epochs 3 --model-dim 96 --tcn-blocks 3 --tail-minutes 30 \
--max-minutes 242 --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --fast-pack"

# main trainer (kernel variants); $1=name $2=seed $3=label $4..=kernels
run_main() {
  local name=$1 seed=$2 lab=$3; shift 3
  local dir=$OUT/$name
  [ -s "$dir/oos_metrics.json" ] && { echo "[skip ] $name"; return; }
  local extra=""; [ "$lab" != "o2c" ] && extra="--label-column ret_open_to_open --extra-labels $O2O"
  echo "[start] $name $(date -Is)"
  if $PY scripts/evaluate_unified_microstructure.py $COMMON --kernels "$@" \
       --output-dir "$dir" --seed "$seed" $extra > "$OUT/$name.log" 2>&1
  then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}
# ablation trainer; $1=name $2=seed $3=label $4=disable-flag
run_abl() {
  local name=$1 seed=$2 lab=$3 flag=$4
  local dir=$OUT/$name
  [ -s "$dir/oos_metrics.json" ] && { echo "[skip ] $name"; return; }
  local extra=""; [ "$lab" != "o2c" ] && extra="--label-column ret_open_to_open --extra-labels $O2O"
  echo "[start] $name $(date -Is)"
  if $PY "$ABL" $COMMON --kernels 3 15 60 --output-dir "$dir" --seed "$seed" "$flag" $extra \
       > "$OUT/$name.log" 2>&1
  then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}

lane_a() {   # P1: E3 family trained on o2o (full arm already exists as E6b)
  for s in 20260801 20260812 20260823; do
    run_abl "o2o_seq_only_$s"    "$s" o2o --disable-statistics-path
    run_abl "o2o_stats_only_$s"  "$s" o2o --disable-sequence-path
    run_abl "o2o_no_deepsets_$s" "$s" o2o --disable-cross-section
  done
  echo "[LANE A DONE] $(date -Is)"
}
lane_b() {   # P2 + kernel seeds
  for s in 20260801 20260812 20260823; do run_main "o2o_k21045_$s" "$s" o2o 2 10 45; done
  run_main o2c_k21045_20260823 20260823 o2c 2 10 45
  for s in 20260812 20260823; do run_main "o2c_k53012_$s" "$s" o2c 5 30 120; done
  echo "[LANE B DONE] $(date -Is)"
}
lane_c() {   # P3: single-scale + E3 family extra seeds on o2c
  for s in 20260812 20260823; do
    run_main "o2c_k3_$s"  "$s" o2c 3
    run_main "o2c_k15_$s" "$s" o2c 15
    run_main "o2c_k60_$s" "$s" o2c 60
    run_abl  "o2c_seq_only_$s"    "$s" o2c --disable-statistics-path
    run_abl  "o2c_stats_only_$s"  "$s" o2c --disable-sequence-path
    run_abl  "o2c_no_deepsets_$s" "$s" o2c --disable-cross-section
  done
  echo "[LANE C DONE] $(date -Is)"
}
lane_a > "$OUT/lane_a.log" 2>&1 &
lane_b > "$OUT/lane_b.log" 2>&1 &
lane_c > "$OUT/lane_c.log" 2>&1 &
wait
echo "[ALL LANES DONE] $(date -Is)"
